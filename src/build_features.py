"""
Build team-level and squad-level features from:
  - Historical international results (data/raw/international_results/results.csv)
  - Parsed squad PDF (data/processed/squads_parsed.csv)
  - Kaggle player-scores dataset (data/raw/player_scores/*.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz, utils
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, PROCESSED_DIR, MODEL_CONFIG_PATH, KAGGLE_DIR
from src.normalize_teams import normalize_team_name
import yaml


# ─── helpers ─────────────────────────────────────────────────────────────────

def load_model_config() -> dict:
    if MODEL_CONFIG_PATH.exists():
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ─── historical team features ───────────────────────────────────────────────

def calculate_team_features() -> pd.DataFrame:
    """Build per-team attack/defense features from historical results."""
    results_path = RAW_DIR / "international_results" / "results.csv"
    if not results_path.exists():
        print("[features] results.csv not found – historical features unavailable.")
        return pd.DataFrame()

    df = pd.read_csv(results_path)

    # Normalise team names
    unique_teams = set(df["home_team"]).union(set(df["away_team"]))
    team_map = {t: normalize_team_name(t) for t in unique_teams}
    df["home_team"] = df["home_team"].map(team_map)
    df["away_team"] = df["away_team"].map(team_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    print(f"[features] Loaded {len(df)} historical results.")
    return df


# ─── Kaggle player matching ─────────────────────────────────────────────────

def _load_kaggle_players() -> pd.DataFrame:
    """Load players.csv from the Kaggle dataset."""
    path = KAGGLE_DIR / "players.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df


def _load_kaggle_valuations() -> pd.DataFrame:
    """Load player_valuations.csv and return the latest valuation per player."""
    path = KAGGLE_DIR / "player_valuations.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        # Keep only the most recent valuation per player
        df = df.sort_values("date").groupby("player_id").tail(1)
    return df


def _load_kaggle_appearances() -> pd.DataFrame:
    """Load appearances.csv from Kaggle."""
    path = KAGGLE_DIR / "appearances.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _fuzzy_match_player(squad_name: str, kaggle_names: dict, threshold: float = 85.0):
    """Fuzzy-match a squad player name against Kaggle player names using token_set_ratio.
       kaggle_names is expected to be a dictionary mapping the original name to its processed form,
       so we can use process.extractOne efficiently."""
    if not kaggle_names:
        return None, 0.0
    
    # Create list of pre-processed names or let extractOne use the dict keys with processor.
    # Passing processor=utils.default_process handles lowercasing and non-alphanumeric removal
    match = process.extractOne(
        squad_name, 
        kaggle_names, 
        scorer=fuzz.token_set_ratio, 
        processor=utils.default_process,
        score_cutoff=threshold
    )
    if match:
        return match[0], match[1]
    return None, 0.0


def match_squad_to_kaggle(squads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to match every squad player to a Kaggle player record.
    Returns the squads dataframe enriched with Kaggle columns.
    Also writes a matching report to data/processed/.
    """
    kg_players = _load_kaggle_players()
    kg_valuations = _load_kaggle_valuations()
    kg_appearances = _load_kaggle_appearances()

    if kg_players.empty:
        print("[features] WARNING: Kaggle players.csv not available – skipping player matching.")
        squads_df["kaggle_player_id"] = None
        squads_df["kaggle_match_score"] = 0.0
        squads_df["market_value_eur"] = np.nan
        squads_df["recent_appearances"] = np.nan
        squads_df["recent_goals"] = np.nan
        squads_df["recent_assists"] = np.nan
        return squads_df

    # Build lookup: use "name" column from Kaggle (fallback to "pretty_name" or similar)
    name_col = "name" if "name" in kg_players.columns else "pretty_name" if "pretty_name" in kg_players.columns else None
    if name_col is None:
        print("[features] WARNING: Kaggle players.csv has no name column – cannot match.")
        squads_df["kaggle_player_id"] = None
        squads_df["kaggle_match_score"] = 0.0
        squads_df["market_value_eur"] = np.nan
        squads_df["recent_appearances"] = np.nan
        squads_df["recent_goals"] = np.nan
        squads_df["recent_assists"] = np.nan
        return squads_df

    kg_players_clean = kg_players.dropna(subset=[name_col]).copy()
    # Build dict: kaggle_name → player_id
    name_to_id = dict(zip(kg_players_clean[name_col], kg_players_clean["player_id"]))
    kaggle_names = list(name_to_id.keys())

    # Valuation lookup
    val_lookup = {}
    if not kg_valuations.empty and "player_id" in kg_valuations.columns and "market_value_in_eur" in kg_valuations.columns:
        val_lookup = dict(zip(kg_valuations["player_id"], kg_valuations["market_value_in_eur"]))

    # Recent appearances: last-season stats
    app_goals = {}
    app_assists = {}
    app_counts = {}
    if not kg_appearances.empty:
        if "date" in kg_appearances.columns:
            kg_appearances["date"] = pd.to_datetime(kg_appearances["date"], errors="coerce")
            # Filter to last ~18 months
            cutoff = pd.Timestamp.now() - pd.DateOffset(months=18)
            recent = kg_appearances[kg_appearances["date"] >= cutoff]
        else:
            recent = kg_appearances

        if "player_id" in recent.columns:
            grouped = recent.groupby("player_id")
            app_counts = grouped.size().to_dict()
            if "goals" in recent.columns:
                app_goals = grouped["goals"].sum().to_dict()
            if "assists" in recent.columns:
                app_assists = grouped["assists"].sum().to_dict()

    # ── Match each squad player ──────────────────────────────────────────
    matched_ids = []
    matched_scores = []
    market_values = []
    recent_apps_list = []
    recent_goals_list = []
    recent_assists_list = []

    # We try to extract a clean player name from the squad's player_name column.
    # The PDF parser produces names like "MANDI Aissa Aissa MANDI MANDI".
    # Strategy: use shirt_name as main matching key.
    for _, row in squads_df.iterrows():
        # Prefer shirt_name for matching as it's shorter and cleaner
        search_name = str(row.get("shirt_name", row.get("player_name", "")))
        player_full = str(row.get("player_name", ""))

        # Try shirt_name first, then full name
        best_name, best_score = _fuzzy_match_player(search_name, kaggle_names, threshold=85.0)
        if best_score < 85.0 and player_full != search_name:
            alt_name, alt_score = _fuzzy_match_player(player_full, kaggle_names, threshold=85.0)
            if alt_score > best_score:
                best_name, best_score = alt_name, alt_score

        if best_name:
            pid = name_to_id.get(best_name)
            matched_ids.append(pid)
            matched_scores.append(best_score)
            market_values.append(val_lookup.get(pid, np.nan))
            recent_apps_list.append(app_counts.get(pid, np.nan))
            recent_goals_list.append(app_goals.get(pid, np.nan))
            recent_assists_list.append(app_assists.get(pid, np.nan))
        else:
            matched_ids.append(None)
            matched_scores.append(0.0)
            market_values.append(np.nan)
            recent_apps_list.append(np.nan)
            recent_goals_list.append(np.nan)
            recent_assists_list.append(np.nan)

    squads_df["kaggle_player_id"] = matched_ids
    squads_df["kaggle_match_score"] = matched_scores
    squads_df["market_value_eur"] = market_values
    squads_df["recent_appearances"] = recent_apps_list
    squads_df["recent_goals"] = recent_goals_list
    squads_df["recent_assists"] = recent_assists_list

    # ── Write matching report & detailed logging ─────────────────────────
    total = len(squads_df)
    matched = squads_df["kaggle_player_id"].notna().sum()
    rate = matched / total if total > 0 else 0.0
    print(f"[features] Player matching overall: {matched}/{total} matched ({rate:.1%})")

    # Group by team and log missing players
    for team, grp in squads_df.groupby("team"):
        t_total = len(grp)
        t_matched = grp["kaggle_player_id"].notna().sum()
        t_rate = t_matched / t_total if t_total > 0 else 0.0
        unmatched = grp[grp["kaggle_player_id"].isna()]["player_name"].tolist()
        
        print(f"[Matching] {team}: {t_matched}/{t_total} jugadores ({t_rate:.1%})")
        if unmatched:
            # Safely encode to avoid UnicodeEncodeError in Windows terminal
            safe_unmatched = [name.encode('ascii', 'replace').decode('ascii') for name in unmatched[:5]]
            print(f"    No matcheados: {safe_unmatched}{'...' if len(unmatched) > 5 else ''}")

    report_path = PROCESSED_DIR / "player_matching_report.csv"
    report_cols = ["team", "player_name", "shirt_name", "kaggle_player_id",
                   "kaggle_match_score", "market_value_eur"]
    squads_df[report_cols].to_csv(report_path, index=False)
    print(f"[features] Matching report saved to {report_path}")

    return squads_df


# ─── squad-level aggregate features ─────────────────────────────────────────

def calculate_squad_features() -> pd.DataFrame:
    """
    Build per-team squad strength features by merging:
      - Parsed squad PDF  (caps, goals, position, DOB, club)
      - Kaggle player data (market value, recent appearances/goals/assists)
    """
    squads_path = PROCESSED_DIR / "squads_parsed.csv"
    if not squads_path.exists():
        print("[features] squads_parsed.csv not found – run PDF parser first.")
        return pd.DataFrame()

    squads = pd.read_csv(squads_path)
    if squads.empty:
        print("[features] squads_parsed.csv is empty.")
        return pd.DataFrame()

    # Enrich with Kaggle data
    squads = match_squad_to_kaggle(squads)

    # Calculate player age
    if "dob" in squads.columns:
        squads["dob_parsed"] = pd.to_datetime(squads["dob"], format="%d/%m/%Y", errors="coerce")
        today = pd.Timestamp.now()
        squads["age"] = ((today - squads["dob_parsed"]).dt.days / 365.25).round(1)

    # ── Aggregate per team ───────────────────────────────────────────────
    team_features = []
    for team, grp in squads.groupby("team"):
        n_players = len(grp)
        kaggle_matched = int(grp["kaggle_player_id"].notna().sum())
        match_rate = kaggle_matched / n_players if n_players > 0 else 0.0

        # Market values
        mv = grp["market_value_eur"].dropna()
        total_mv = float(mv.sum()) if not mv.empty else 0.0
        avg_mv = float(mv.mean()) if not mv.empty else 0.0
        sorted_mv = mv.sort_values(ascending=False)
        top11_mv = float(sorted_mv.head(11).mean()) if len(sorted_mv) >= 11 else float(sorted_mv.mean()) if not sorted_mv.empty else 0.0
        top18_mv = float(sorted_mv.head(18).mean()) if len(sorted_mv) >= 18 else float(sorted_mv.mean()) if not sorted_mv.empty else 0.0

        # Position-based value distribution
        pos_map = {"PO": "goalkeeper", "DF": "defender", "MC": "midfielder", "DC": "forward"}
        pos_values = {}
        for pos_code, pos_label in pos_map.items():
            pos_grp = grp[grp["position"] == pos_code]["market_value_eur"].dropna()
            pos_values[f"{pos_label}_total_value"] = float(pos_grp.sum()) if not pos_grp.empty else 0.0

        # Squad Quality Index (SQI) Ponderado
        def calc_sqi(row):
            val = row.get("market_value_eur")
            if pd.isna(val): return 0
            pos = row.get("position")
            if pos == "DC": return val * 1.5
            if pos == "MC": return val * 1.2
            return val * 1.0
            
        grp_copy = grp.copy()
        grp_copy["weighted_value"] = grp_copy.apply(calc_sqi, axis=1)
        sqi_raw_sum = grp_copy["weighted_value"].sum()
        
        # Age
        ages = grp["age"].dropna() if "age" in grp.columns else pd.Series(dtype=float)
        avg_age = float(ages.mean()) if not ages.empty else np.nan

        # Caps / goals from PDF
        total_caps = int(grp["caps"].sum()) if "caps" in grp.columns else 0
        total_goals = int(grp["goals"].sum()) if "goals" in grp.columns else 0
        fwd_goals = int(grp.loc[grp["position"] == "DC", "goals"].sum()) if "goals" in grp.columns else 0
        gk_caps = int(grp.loc[grp["position"] == "PO", "caps"].sum()) if "caps" in grp.columns else 0
        def_caps = int(grp.loc[grp["position"] == "DF", "caps"].sum()) if "caps" in grp.columns else 0

        # Recent club form from Kaggle
        recent_apps = grp["recent_appearances"].dropna()
        recent_goals_s = grp["recent_goals"].dropna()
        recent_assists_s = grp["recent_assists"].dropna()
        
        total_recent_apps = int(recent_apps.sum()) if not recent_apps.empty else 0
        total_recent_goals = int(recent_goals_s.sum()) if not recent_goals_s.empty else 0
        
        # PROJECTION FOR POOR MATCHING
        # If match rate is < 1.0 but > 0, project the values
        proj_factor = 1.0 / match_rate if 0 < match_rate < 1.0 else 1.0
        
        total_mv_proj = total_mv * proj_factor
        sqi_proj = sqi_raw_sum * proj_factor
        recent_goals_proj = total_recent_goals * proj_factor
        recent_apps_proj = total_recent_apps * proj_factor
        
        club_goals_per_app = round(recent_goals_proj / recent_apps_proj, 3) if recent_apps_proj > 0 else 0.0

        feat = {
            "team": team,
            "squad_size": n_players,
            "kaggle_match_count": kaggle_matched,
            "kaggle_match_rate": round(match_rate, 3),
            "squad_total_market_value": round(total_mv_proj, 0),
            "squad_avg_market_value": round(avg_mv, 0),
            "top11_avg_market_value": round(top11_mv, 0),
            "top18_avg_market_value": round(top18_mv, 0),
            **{k: round(v * proj_factor, 0) for k, v in pos_values.items()},
            "squad_avg_age": round(avg_age, 1) if not np.isnan(avg_age) else None,
            "total_caps": total_caps,
            "total_international_goals": total_goals,
            "forward_international_goals": fwd_goals,
            "goalkeeper_caps": gk_caps,
            "defensive_experience_caps": def_caps,
            "recent_club_appearances_total": round(recent_apps_proj, 1),
            "recent_club_goals_total": round(recent_goals_proj, 1),
            "recent_club_assists_total": round(int(recent_assists_s.sum()) * proj_factor, 1) if not recent_assists_s.empty else 0,
            "squad_quality_index": round(sqi_proj, 0),
            "recent_club_goals_per_app": club_goals_per_app,
            "data_quality_warnings": [],
        }

        # Warnings
        if match_rate < 0.5:
            feat["data_quality_warnings"].append(
                f"Low Kaggle match rate ({match_rate:.0%}) – squad features mathematically projected but may be unreliable"
            )
        if total_mv == 0:
            feat["data_quality_warnings"].append("No market value data available")

        team_features.append(feat)

    df_features = pd.DataFrame(team_features)
    return df_features


# ─── main entry point ───────────────────────────────────────────────────────

def build_features():
    """Build and persist all feature sets."""
    # 1. Historical team features
    team_hist = calculate_team_features()
    if not team_hist.empty:
        out = PROCESSED_DIR / "historical_features.csv"
        team_hist.to_csv(out, index=False)
        print(f"[features] Saved historical features -> {out}")

    # 2. Squad features (PDF + Kaggle)
    squad_feat = calculate_squad_features()
    if not squad_feat.empty:
        # Convert warnings list to string for CSV
        squad_feat["data_quality_warnings"] = squad_feat["data_quality_warnings"].apply(
            lambda w: "; ".join(w) if isinstance(w, list) else ""
        )
        out = PROCESSED_DIR / "squad_features.csv"
        squad_feat.to_csv(out, index=False)
        print(f"[features] Saved squad features -> {out}")

        # Summary
        avg_rate = squad_feat["kaggle_match_rate"].mean()
        total_val = squad_feat["squad_total_market_value"].sum()
        print(f"[features] {len(squad_feat)} teams processed")
        print(f"[features] Average Kaggle match rate: {avg_rate:.1%}")
        print(f"[features] Total squad market value: EUR {total_val:,.0f}")
    else:
        print("[features] No squad features generated.")


if __name__ == "__main__":
    build_features()
