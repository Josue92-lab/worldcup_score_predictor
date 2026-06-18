import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIXTURE_PATH, VENUES_PATH, AUDIT_REPORT_PATH,
    KAGGLE_DIR, KAGGLE_ZIP_PATH, KAGGLE_EXPECTED_CSVS,
    PROCESSED_DIR, RAW_DIR
)
from src.normalize_teams import normalize_team_name
from src.model_poisson import fit_poisson_model

# Known knockout placeholders roughly
KNOCKOUT_PLACEHOLDERS = [
    "Winner", "Runner-up", "Third place", "Loser", "/"
]

def is_knockout_placeholder(team_name):
    if not isinstance(team_name, str):
        return False
    for placeholder in KNOCKOUT_PLACEHOLDERS:
        if placeholder in team_name:
            return True
    return False


def _audit_kaggle() -> dict:
    """Audit the Kaggle player-scores dataset files."""
    section = {
        "download_method": "direct_http_zip",
        "zip_path": str(KAGGLE_ZIP_PATH.relative_to(KAGGLE_DIR.parent.parent.parent)),
        "extract_path": str(KAGGLE_DIR.relative_to(KAGGLE_DIR.parent.parent.parent)),
        "expected_files": list(KAGGLE_EXPECTED_CSVS),
        "missing_files": [],
        "file_row_counts": {},
        "status": "fail",
    }

    if not KAGGLE_DIR.exists():
        section["missing_files"] = list(KAGGLE_EXPECTED_CSVS)
        return section

    missing = []
    for csv_name in KAGGLE_EXPECTED_CSVS:
        csv_path = KAGGLE_DIR / csv_name
        if csv_path.exists():
            try:
                row_count = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
                section["file_row_counts"][csv_name] = max(row_count, 0)
            except Exception:
                section["file_row_counts"][csv_name] = -1
        else:
            missing.append(csv_name)

    section["missing_files"] = missing

    if not missing:
        section["status"] = "pass"
    elif len(missing) < len(KAGGLE_EXPECTED_CSVS):
        section["status"] = "needs_review"
    # else stays "fail"

    return section


def _audit_squad_to_fixture(fixture_df, squads_path) -> tuple:
    """Cross-reference squad teams with fixture teams."""
    unmatched_fixture_to_squad = []
    unmatched_squad_to_fixture = []

    if not squads_path.exists():
        return unmatched_fixture_to_squad, unmatched_squad_to_fixture

    squads = pd.read_csv(squads_path)
    squad_teams = set(squads["team"].dropna().unique()) if "team" in squads.columns else set()

    if not fixture_df.empty:
        all_fixture_teams = set()
        for col in ["home_team", "away_team"]:
            if col in fixture_df.columns:
                raw_teams = fixture_df[col].dropna().unique()
                for t in raw_teams:
                    if not is_knockout_placeholder(t):
                        all_fixture_teams.add(normalize_team_name(t))

        unmatched_fixture_to_squad = sorted(all_fixture_teams - squad_teams)
        unmatched_squad_to_fixture = sorted(squad_teams - all_fixture_teams)

    return unmatched_fixture_to_squad, unmatched_squad_to_fixture

def _build_team_lineage(fixture_df: pd.DataFrame, squads_df: pd.DataFrame):
    """
    Generate data/audit/team_feature_lineage.csv and .json with one row per team
    to explain why some teams might have missing attack/defense/squad data.
    """
    lineage_path_csv = AUDIT_REPORT_PATH.parent / "team_feature_lineage.csv"
    lineage_path_json = AUDIT_REPORT_PATH.parent / "team_feature_lineage.json"

    # 1. Get fixture teams
    fixture_teams = set()
    if not fixture_df.empty:
        for col in ["home_team", "away_team"]:
            if col in fixture_df.columns:
                raw_teams = fixture_df[col].dropna().unique()
                for t in raw_teams:
                    if not is_knockout_placeholder(t):
                        fixture_teams.add(normalize_team_name(t))
    
    # 2. Get squad teams
    squad_teams_map = {}
    if not squads_df.empty and "team" in squads_df.columns:
        squads_df["team_norm"] = squads_df["team"].apply(normalize_team_name)
        for t, grp in squads_df.groupby("team_norm"):
            squad_teams_map[t] = len(grp)
            
    # 3. Get Kaggle match rates
    kaggle_report_path = PROCESSED_DIR / "player_matching_report.csv"
    kaggle_map = {}
    if kaggle_report_path.exists():
        kg = pd.read_csv(kaggle_report_path)
        if "team" in kg.columns and "kaggle_player_id" in kg.columns:
            kg["team_norm"] = kg["team"].apply(normalize_team_name)
            for t, grp in kg.groupby("team_norm"):
                n_players = len(grp)
                matched = grp["kaggle_player_id"].notna().sum()
                rate = matched / n_players if n_players > 0 else 0.0
                kaggle_map[t] = {
                    "matched": matched,
                    "rate": rate,
                    "squad_strength": float(grp.get("market_value_eur", pd.Series(dtype=float)).sum())
                }

    # 4. Get historical results & model
    hist_path = RAW_DIR / "international_results" / "results.csv"
    model = {"attack": {}, "defense": {}}
    hist_match_counts = {}
    hist_match_counts_before_cutoff = {}
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        hist_df["home_team_norm"] = hist_df["home_team"].apply(normalize_team_name)
        hist_df["away_team_norm"] = hist_df["away_team"].apply(normalize_team_name)
        
        # Count total
        for t in pd.concat([hist_df["home_team_norm"], hist_df["away_team_norm"]]):
            hist_match_counts[t] = hist_match_counts.get(t, 0) + 1
            
        # Count before cutoff (simulating prediction cutoff logic: 2026-06-10)
        hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
        cutoff = pd.to_datetime("2026-06-10")
        hist_before = hist_df[hist_df["date"] <= cutoff].copy()
        for t in pd.concat([hist_before["home_team_norm"], hist_before["away_team_norm"]]):
            hist_match_counts_before_cutoff[t] = hist_match_counts_before_cutoff.get(t, 0) + 1
            
        # Fit model so we get EXACT attack/defense
        # To fit model_poisson we need home_team, away_team, home_score, away_score
        hist_before["home_team"] = hist_before["home_team_norm"]
        hist_before["away_team"] = hist_before["away_team_norm"]
        model = fit_poisson_model(hist_before)

    # Compile lineage
    all_teams = fixture_teams.copy()
    
    lineage_data = []
    for t in sorted(list(all_teams)):
        in_fixture = t in fixture_teams
        squad_players = squad_teams_map.get(t, 0)
        in_squad = squad_players > 0
        
        kg = kaggle_map.get(t, {})
        kg_matched = kg.get("matched", 0)
        kg_rate = kg.get("rate", 0.0)
        sq_strength = kg.get("squad_strength", 0.0)
        
        hist_matches = hist_match_counts.get(t, 0)
        hist_cutoff = hist_match_counts_before_cutoff.get(t, 0)
        
        att = model["attack"].get(t)
        dfn = model["defense"].get(t)
        
        fallback = att is None or dfn is None
        
        missing_reason = ""
        warnings = []
        
        if fallback:
            if hist_matches == 0:
                missing_reason = "No rows in martj42 results.csv - potential name mismatch or no history."
            elif hist_cutoff == 0:
                missing_reason = "Matches exist but none before cutoff (2026-06-10)."
            else:
                missing_reason = "Model failed to generate attack/defense despite matches existing."
                
        if not in_squad:
            warnings.append("No squad data found in PDF.")
        if kg_rate < 0.5:
            warnings.append(f"Low Kaggle match rate ({kg_rate:.1%}).")
            
        lineage_data.append({
            "team": t,
            "canonical_team": t,
            "fixture_present": bool(in_fixture),
            "squad_present": bool(in_squad),
            "squad_players": int(squad_players),
            "kaggle_matched_players": int(kg_matched),
            "kaggle_match_rate": float(round(kg_rate, 3)),
            "historical_results_matches": int(hist_matches),
            "historical_results_matches_before_cutoff": int(hist_cutoff),
            "attack_strength": float(round(att, 4)) if att is not None else None,
            "defense_strength": float(round(dfn, 4)) if dfn is not None else None,
            "squad_strength": float(sq_strength),
            "fallback_used": bool(fallback),
            "missing_reason": str(missing_reason),
            "warnings": "; ".join(warnings)
        })
        
    df_lin = pd.DataFrame(lineage_data)
    df_lin.to_csv(lineage_path_csv, index=False)
    with open(lineage_path_json, "w", encoding="utf-8") as f:
        json.dump(lineage_data, f, indent=2)
        
    print(f"Team feature lineage saved to {lineage_path_csv}")



def audit_data():
    report = {
        "fixture_rows": 0,
        "venues_rows": 0,
        "squad_teams": 0,
        "squad_players": 0,
        "duplicate_match_ids": [],
        "missing_fixture_values": {},
        "missing_venue_codes": [],
        "invalid_dates": [],
        "unmatched_fixture_teams_to_squads": [],
        "unmatched_squad_teams_to_fixture": [],
        "unmatched_squad_players_to_kaggle": [],
        "team_aliases_required": [],
        "knockout_placeholders_detected": False,
        "kaggle_player_scores": {},
        "warnings": [],
        "status": "needs_review"
    }

    # ── 1. Audit Venues ──────────────────────────────────────────────────
    venues = pd.DataFrame()
    if VENUES_PATH.exists():
        venues = pd.read_csv(VENUES_PATH)
        report["venues_rows"] = len(venues)

        if "venue_code" in venues.columns:
            dupes = venues[venues.duplicated("venue_code", keep=False)]
            if not dupes.empty:
                report["warnings"].append(
                    f"Found duplicate venue codes: {dupes['venue_code'].unique().tolist()}"
                )

        expected_venue_cols = ["venue_code", "city", "stadium", "timezone", "latitude", "longitude"]
        for col in expected_venue_cols:
            if col in venues.columns:
                missing = venues[venues[col].isnull()]
                if not missing.empty:
                    report["warnings"].append(
                        f"venues.csv missing values in {col}: {len(missing)} rows"
                    )
    else:
        report["warnings"].append("venues.csv not found")

    # ── 2. Audit Fixture ─────────────────────────────────────────────────
    fixture = pd.DataFrame()
    if FIXTURE_PATH.exists():
        fixture = pd.read_csv(FIXTURE_PATH)
        report["fixture_rows"] = len(fixture)

        if "match_id" in fixture.columns:
            dupes = fixture[fixture.duplicated("match_id", keep=False)]
            if not dupes.empty:
                report["duplicate_match_ids"] = dupes["match_id"].unique().tolist()

        # Missing values
        missing_dict = {}
        for col in fixture.columns:
            missing_count = int(fixture[col].isnull().sum())
            if missing_count > 0:
                missing_dict[col] = missing_count
        report["missing_fixture_values"] = missing_dict

        # Invalid dates
        f_val = {
            "world_cup_start": "2026-06-11",
            "world_cup_end": "2026-07-19",
            "matches_before_start": [],
            "matches_after_end": [],
            "group_stage_after_knockout_started": [],
            "suspicious_local_kickoffs": [],
            "status": "pass"
        }
        report["fixture_date_validation"] = f_val

        if "date" in fixture.columns:
            try:
                parsed_dates = pd.to_datetime(fixture["date"], errors="coerce")
                invalid_dates = fixture[parsed_dates.isnull()]["date"].dropna().tolist()
                report["invalid_dates"] = invalid_dates
            except Exception as e:
                report["warnings"].append(f"Error parsing dates: {e}")

            # Bounds and timezone checking
            if "time_utc" in fixture.columns and "venue_code" in fixture.columns:
                wc_start = pd.to_datetime(f_val["world_cup_start"])
                wc_end = pd.to_datetime(f_val["world_cup_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                valid_dt_mask = parsed_dates.notnull()
                df_valid = fixture[valid_dt_mask].copy()
                df_valid["dt"] = pd.to_datetime(df_valid["date"] + " " + df_valid["time_utc"], errors="coerce")
                df_valid = df_valid.dropna(subset=["dt"])

                # Check bounds
                before = df_valid[df_valid["dt"] < wc_start]
                if not before.empty:
                    f_val["matches_before_start"] = before["match_id"].tolist()
                
                after = df_valid[df_valid["dt"] > wc_end]
                if not after.empty:
                    f_val["matches_after_end"] = after["match_id"].tolist()

                # Group vs Knockout
                ko_mask = df_valid["stage"].str.contains("Round of|Quarter-final|Semi-final|Third place|Final", na=False, case=False)
                knockouts = df_valid[ko_mask]
                if not knockouts.empty:
                    first_ko_dt = knockouts["dt"].min()
                    gs_mask = df_valid["stage"].str.contains("Group", na=False, case=False)
                    bad_gs = df_valid[gs_mask & (df_valid["dt"] > first_ko_dt)]
                    if not bad_gs.empty:
                        f_val["group_stage_after_knockout_started"] = bad_gs["match_id"].tolist()

                # Timezone checking
                venue_tzs = {}
                if not venues.empty and "venue_code" in venues.columns and "timezone" in venues.columns:
                    venue_tzs = dict(zip(venues["venue_code"], venues["timezone"]))
                
                suspicious = []
                for _, row in df_valid.iterrows():
                    vc = row.get("venue_code")
                    tz_str = venue_tzs.get(vc)
                    if tz_str:
                        try:
                            dt_utc = row["dt"].tz_localize("UTC")
                            dt_local = dt_utc.tz_convert(tz_str)
                            hr = dt_local.hour + dt_local.minute / 60.0
                            if hr < 9.0 or hr > 23.5:
                                suspicious.append(row["match_id"])
                        except Exception:
                            pass
                
                if suspicious:
                    f_val["suspicious_local_kickoffs"] = suspicious
                
                if f_val["matches_before_start"] or f_val["matches_after_end"] or f_val["group_stage_after_knockout_started"]:
                    f_val["status"] = "fail"
                elif f_val["suspicious_local_kickoffs"]:
                    f_val["status"] = "needs_review"

        # Knockout Placeholders
        teams = pd.concat(
            [fixture.get("home_team", pd.Series()), fixture.get("away_team", pd.Series())]
        ).dropna().unique()
        placeholders = [t for t in teams if is_knockout_placeholder(t)]
        if placeholders:
            report["knockout_placeholders_detected"] = True
            report["warnings"].append(
                f"Knockout placeholders found: {len(placeholders)} unique placeholders."
            )

        # Venue codes missing
        if "venue_code" in fixture.columns and not venues.empty and "venue_code" in venues.columns:
            venue_codes_fixture = fixture["venue_code"].dropna().unique()
            venue_codes_venues = venues["venue_code"].dropna().unique()
            missing_vc = [vc for vc in venue_codes_fixture if vc not in venue_codes_venues]
            report["missing_venue_codes"] = missing_vc
    else:
        report["warnings"].append("fixture.csv not found")

    # ── 3. Audit Squads ──────────────────────────────────────────────────
    squads_path = PROCESSED_DIR / "squads_parsed.csv"
    if squads_path.exists():
        squads = pd.read_csv(squads_path)
        report["squad_teams"] = int(squads["team"].nunique()) if "team" in squads.columns else 0
        report["squad_players"] = len(squads)

        unmatched_f2s, unmatched_s2f = _audit_squad_to_fixture(fixture, squads_path)
        report["unmatched_fixture_teams_to_squads"] = unmatched_f2s
        report["unmatched_squad_teams_to_fixture"] = unmatched_s2f
    else:
        report["warnings"].append("squads_parsed.csv not found – run PDF parser first")

    # ── 4. Audit Kaggle player-scores ────────────────────────────────────
    kaggle_section = _audit_kaggle()
    report["kaggle_player_scores"] = kaggle_section

    if kaggle_section["status"] == "fail":
        report["warnings"].append(
            "Kaggle player-scores dataset is missing or incomplete. "
            "Run: python -m src.cli download-kaggle"
        )

    # ── 5. Derive overall status ─────────────────────────────────────────
    critical_failures = (
        len(report["duplicate_match_ids"]) > 0
        or len(report["invalid_dates"]) > 0
        or kaggle_section["status"] == "fail"
    )
    if critical_failures:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "needs_review"
    else:
        report["status"] = "pass"

    # ── Save Report ──────────────────────────────────────────────────────
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Audit report saved to {AUDIT_REPORT_PATH}")
    print(f"Overall status: {report['status']}")
    print(f"Kaggle player-scores status: {kaggle_section['status']}")

    if report["warnings"]:
        print(f"\n{len(report['warnings'])} warning(s):")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    # ── 6. Build Data Lineage Report ─────────────────────────────────────
    squads_for_lineage = pd.DataFrame()
    if squads_path.exists():
        squads_for_lineage = pd.read_csv(squads_path)
    _build_team_lineage(fixture, squads_for_lineage)

    return report


if __name__ == "__main__":
    audit_data()
