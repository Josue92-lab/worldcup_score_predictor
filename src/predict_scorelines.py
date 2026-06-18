"""
Predict top-5 scorelines for each fixture using the Poisson model
and real feature data where available.
"""

import pandas as pd
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIXTURE_PATH, PROCESSED_DIR, PREDICTIONS_CSV_PATH, PREDICTIONS_JSON_PATH
from src.model_poisson import predict_xg, calculate_match_probabilities, fit_poisson_model
from src.normalize_teams import normalize_team_name
from src.build_features import load_model_config
from src.audit_data import is_knockout_placeholder


def _load_squad_features() -> dict:
    """Load squad features keyed by normalised team name."""
    path = PROCESSED_DIR / "squad_features.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "team" not in df.columns:
        return {}
    return {row["team"]: row.to_dict() for _, row in df.iterrows()}


def predict_scorelines(mode="live", as_of_date=None, train_cutoff=None):
    if not FIXTURE_PATH.exists():
        print(f"Error: Fixture file {FIXTURE_PATH} not found.")
        return

    fixtures = pd.read_csv(FIXTURE_PATH)
    config = load_model_config()
    max_goals = config.get("model", {}).get("max_goals", 7)

    # ── Load historical results and fit the Poisson model ────────────────
    hist_path = PROCESSED_DIR / "historical_features.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        
        # Filter historical data based on mode
        hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
        if mode == "backtest" and train_cutoff:
            cutoff_dt = pd.to_datetime(train_cutoff)
            hist_df = hist_df[hist_df["date"] <= cutoff_dt]
        elif mode == "live" and as_of_date:
            asof_dt = pd.to_datetime(as_of_date)
            hist_df = hist_df[hist_df["date"] <= asof_dt]
            
        print(f"[predict] Fitting Poisson model from {len(hist_df)} historical results...")
        model = fit_poisson_model(hist_df)
        print(f"[predict] Base goal rate: {model['base_rate']:.3f}")
        print(f"[predict] Teams with attack/defense data: {len(model['attack'])}")
    else:
        print("[predict] WARNING: No historical features – using flat baseline.")
        model = {
            "attack": {},
            "defense": {},
            "base_rate": config.get("model", {}).get("base_rate", 1.2)
        }

    # ── Load squad features ──────────────────────────────────────────────
    squad_feats = _load_squad_features()
    if squad_feats:
        print(f"[predict] Squad features loaded for {len(squad_feats)} teams.")
    else:
        print("[predict] WARNING: No squad features available.")

    # ── Predict each match ───────────────────────────────────────────────
    predictions_json = []

    for idx, row in fixtures.iterrows():
        match_id = row.get("match_id", f"M_{idx}")
        team_a_raw = row.get("home_team", "")
        team_b_raw = row.get("away_team", "")

        # Skip knockout placeholders – no real teams to predict
        if is_knockout_placeholder(team_a_raw) or is_knockout_placeholder(team_b_raw):
            continue

        team_a = normalize_team_name(team_a_raw)
        team_b = normalize_team_name(team_b_raw)

        xg_a, xg_b = predict_xg(team_a, team_b, model)
        probs = calculate_match_probabilities(xg_a, xg_b, max_goals)

        # Squad info
        sq_a = squad_feats.get(team_a, {})
        sq_b = squad_feats.get(team_b, {})

        squad_status_a = "ok" if sq_a.get("kaggle_match_rate", 0) > 0.5 else "partial" if sq_a else "missing"
        squad_status_b = "ok" if sq_b.get("kaggle_match_rate", 0) > 0.5 else "partial" if sq_b else "missing"
        overall_squad_status = "ok"
        if "missing" in (squad_status_a, squad_status_b):
            overall_squad_status = "missing"
        elif "partial" in (squad_status_a, squad_status_b):
            overall_squad_status = "partial"

        dq_warnings = []
        if squad_status_a != "ok":
            dq_warnings.append(f"{team_a}: squad data is {squad_status_a}")
        if squad_status_b != "ok":
            dq_warnings.append(f"{team_b}: squad data is {squad_status_b}")
        if team_a not in model["attack"]:
            dq_warnings.append(f"{team_a}: no historical attack/defense data")
        if team_b not in model["attack"]:
            dq_warnings.append(f"{team_b}: no historical attack/defense data")

        kaggle_rate = min(
            sq_a.get("kaggle_match_rate", 0.0),
            sq_b.get("kaggle_match_rate", 0.0),
        )

        pred_dict = {
            "match_id": match_id,
            "date": row.get("date", ""),
            "phase": row.get("stage", ""),
            "group": row.get("stage", ""),
            "team_a": team_a,
            "team_b": team_b,
            "home_team": team_a,
            "away_team": team_b,
            "team_a_label": "Team A / Listed first",
            "team_b_label": "Team B / Listed second",
            "venue": row.get("venue_code", ""),
            "neutral": True,
            "expected_goals_team_a": round(xg_a, 3),
            "expected_goals_team_b": round(xg_b, 3),
            "team_a_win_probability": round(probs["win_a"], 4),
            "draw_probability": round(probs["draw"], 4),
            "team_b_win_probability": round(probs["win_b"], 4),
            "top_5_scorelines": [
                {
                    "team_a_goals": int(s["scoreline"].split('-')[0]),
                    "team_b_goals": int(s["scoreline"].split('-')[1]),
                    "scoreline": s["scoreline"],
                    "display_scoreline": f"{team_a} {s['scoreline'].split('-')[0]} - {s['scoreline'].split('-')[1]} {team_b}",
                    "probability": round(s["probability"], 4)
                }
                for s in probs["top_5"]
            ],
            "data_quality": {
                "squad_match_status": overall_squad_status,
                "player_kaggle_match_rate": round(kaggle_rate, 3),
                "warnings": dq_warnings,
            },
            "explanation": {
                "team_a_attack_strength": round(model["attack"].get(team_a, 1.0), 4),
                "team_b_attack_strength": round(model["attack"].get(team_b, 1.0), 4),
                "team_a_defense_strength": round(model["defense"].get(team_a, 1.0), 4),
                "team_b_defense_strength": round(model["defense"].get(team_b, 1.0), 4),
                "team_a_squad_strength": round(sq_a.get("squad_total_market_value", 0.0), 0),
                "team_b_squad_strength": round(sq_b.get("squad_total_market_value", 0.0), 0),
                "main_factors": _build_explanation_factors(team_a, team_b, model, sq_a, sq_b),
            },
        }
        predictions_json.append(pred_dict)

    # ── Save Model Params ────────────────────────────────────────────────
    MODEL_PARAMS_PATH = PREDICTIONS_JSON_PATH.parent / "model_params.json"
    with open(MODEL_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"[predict] Saved model params -> {MODEL_PARAMS_PATH}")

    # ── Save JSON ────────────────────────────────────────────────────────
    out_path = PREDICTIONS_JSON_PATH.parent / (
        "backtest_report.json" if mode == "backtest" else "live_predictions.json"
    )
    
    output_obj = {
        "metadata": {
            "mode": mode,
            "as_of_date": as_of_date,
            "train_cutoff": train_cutoff
        },
        "predictions": predictions_json
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)
    print(f"[predict] Saved {len(predictions_json)} predictions -> {out_path}")
    
    # Update PREDICTIONS_CSV_PATH based on mode
    out_csv_path = PREDICTIONS_CSV_PATH.parent / (
        "backtest_report.csv" if mode == "backtest" else "live_predictions.csv"
    )

    # ── Save CSV ─────────────────────────────────────────────────────────
    flat_data = []
    for p in predictions_json:
        flat = {
            "match_id": p["match_id"],
            "date": p["date"],
            "phase": p["phase"],
            "team_a": p["team_a"],
            "team_b": p["team_b"],
            "venue": p["venue"],
            "xG_a": p["expected_goals_team_a"],
            "xG_b": p["expected_goals_team_b"],
            "prob_a_win": p["team_a_win_probability"],
            "prob_draw": p["draw_probability"],
            "prob_b_win": p["team_b_win_probability"],
            "squad_match_status": p["data_quality"]["squad_match_status"],
        }
        for i, sl in enumerate(p["top_5_scorelines"]):
            flat[f"scoreline_{i+1}"] = sl["scoreline"]
            flat[f"scoreline_{i+1}_prob"] = sl["probability"]
        flat_data.append(flat)

    df_out = pd.DataFrame(flat_data)
    df_out.to_csv(out_csv_path, index=False)
    print(f"[predict] Saved CSV -> {out_csv_path}")


def _build_explanation_factors(team_a, team_b, model, sq_a, sq_b) -> list:
    """Generate human-readable explanation factors."""
    factors = []
    att_a = model["attack"].get(team_a)
    att_b = model["attack"].get(team_b)

    if att_a is not None:
        if att_a > 1.2:
            factors.append(f"{team_a} has strong historical attack ({att_a:.2f}x avg)")
        elif att_a < 0.8:
            factors.append(f"{team_a} has weak historical attack ({att_a:.2f}x avg)")
    else:
        factors.append(f"{team_a}: no historical data – using baseline")

    if att_b is not None:
        if att_b > 1.2:
            factors.append(f"{team_b} has strong historical attack ({att_b:.2f}x avg)")
        elif att_b < 0.8:
            factors.append(f"{team_b} has weak historical attack ({att_b:.2f}x avg)")
    else:
        factors.append(f"{team_b}: no historical data – using baseline")

    mv_a = sq_a.get("squad_total_market_value", 0)
    mv_b = sq_b.get("squad_total_market_value", 0)
    if mv_a > 0 and mv_b > 0:
        ratio = mv_a / mv_b if mv_b > 0 else 0
        if ratio > 2:
            factors.append(f"{team_a} squad value is {ratio:.1f}x {team_b}")
        elif ratio < 0.5:
            factors.append(f"{team_b} squad value is {1/ratio:.1f}x {team_a}")

    if not factors:
        factors.append("Baseline Poisson model with default strengths")

    return factors


if __name__ == "__main__":
    predict_scorelines()
