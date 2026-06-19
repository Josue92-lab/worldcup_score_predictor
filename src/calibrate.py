"""
Rolling evaluation and live goal calibration.
"""

import pandas as pd
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, RAW_DIR
from src.normalize_teams import normalize_team_name
from src.build_features import load_model_config

def calculate_calibration_factor(as_of_date=None):
    """
    Computes a goal inflation factor by comparing backtest expected goals
    against actuals up to as_of_date.
    Saves rolling metrics to data/outputs/calibration_report.json.
    Returns:
        dict containing calibration metrics (factor, avg predicted, avg actual, etc.)
    """
    config = load_model_config().get("live_goal_calibration", {})
    if not config.get("enabled", False):
        return {"factor": 1.0, "reason": "Disabled in config", "actual_goals_avg": 0.0, "predicted_goals_avg": 0.0}
    
    max_factor = config.get("max_factor", 1.25)
    min_matches = config.get("min_matches_required", 12)
    
    backtest_path = OUTPUTS_DIR / "backtest_report.json"
    if not backtest_path.exists():
        return {"factor": 1.0, "reason": "No backtest_report.json", "actual_goals_avg": 0.0, "predicted_goals_avg": 0.0}
    
    with open(backtest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    predictions = data.get("predictions", [])
    if not predictions:
        return {"factor": 1.0, "reason": "No predictions in backtest", "actual_goals_avg": 0.0, "predicted_goals_avg": 0.0}
        
    results_path = RAW_DIR / "international_results" / "results.csv"
    if not results_path.exists():
        return {"factor": 1.0, "reason": "No actual results.csv", "actual_goals_avg": 0.0, "predicted_goals_avg": 0.0}
        
    actuals = pd.read_csv(results_path)
    actuals["date"] = pd.to_datetime(actuals["date"], errors="coerce")
    
    actuals["home_team_norm"] = actuals["home_team"].apply(normalize_team_name)
    actuals["away_team_norm"] = actuals["away_team"].apply(normalize_team_name)
    
    wc2026_actuals = actuals[
        (actuals["date"] >= pd.to_datetime("2026-06-11")) &
        (actuals["tournament"].str.contains("World Cup", case=False, na=False))
    ].copy()
    
    if as_of_date:
        wc2026_actuals = wc2026_actuals[wc2026_actuals["date"] <= pd.to_datetime(as_of_date)]
        
    if len(wc2026_actuals) < min_matches:
        return {"factor": 1.0, "reason": f"Only {len(wc2026_actuals)} actual matches < {min_matches} required", "actual_goals_avg": 0.0, "predicted_goals_avg": 0.0}
        
    # Match backtest predictions to actuals
    matched_data = []
    
    for p in predictions:
        p_date = pd.to_datetime(p["date"]).date()
        p_team_a = p["team_a"]
        p_team_b = p["team_b"]
        
        match_actual = wc2026_actuals[
            (wc2026_actuals["date"].dt.date == p_date) & 
            (
                ((wc2026_actuals["home_team_norm"] == p_team_a) & (wc2026_actuals["away_team_norm"] == p_team_b)) |
                ((wc2026_actuals["home_team_norm"] == p_team_b) & (wc2026_actuals["away_team_norm"] == p_team_a))
            )
        ]
        
        if not match_actual.empty:
            row = match_actual.iloc[0]
            if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                continue
                
            actual_goals = int(row["home_score"]) + int(row["away_score"])
            pred_xg = float(p.get("expected_goals_team_a", 0.0)) + float(p.get("expected_goals_team_b", 0.0))
            
            # Hit metrics
            if row["home_team_norm"] == p_team_a:
                actual_score_a = int(row["home_score"])
                actual_score_b = int(row["away_score"])
            else:
                actual_score_a = int(row["away_score"])
                actual_score_b = int(row["home_score"])
                
            actual_scoreline = f"{actual_score_a}-{actual_score_b}"
            
            actual_outcome = "1" if actual_score_a > actual_score_b else "X" if actual_score_a == actual_score_b else "2"
            
            top5 = [s["scoreline"] for s in p["top_5_scorelines"]]
            if top5:
                pred_score_a = int(top5[0].split('-')[0])
                pred_score_b = int(top5[0].split('-')[1])
                pred_outcome = "1" if pred_score_a > pred_score_b else "X" if pred_score_a == pred_score_b else "2"
            else:
                pred_outcome = "X"
                
            is_top1 = actual_scoreline == top5[0] if top5 else False
            is_top5 = actual_scoreline in top5
            is_outcome = actual_outcome == pred_outcome
            
            matched_data.append({
                "date": str(p_date),
                "actual_goals": actual_goals,
                "predicted_goals": pred_xg,
                "top1_exact": is_top1,
                "top5_exact": is_top5,
                "outcome_1x2": is_outcome
            })
            
    df = pd.DataFrame(matched_data)
    if df.empty or len(df) < min_matches:
        return {"factor": 1.0, "reason": f"Only {len(df)} matched rows < {min_matches}", "actual_goals_avg": 0.0, "predicted_goals_avg": 0.0}
        
    df["date"] = pd.to_datetime(df["date"])
    
    # Global metrics
    global_actual = df["actual_goals"].mean()
    global_pred = df["predicted_goals"].mean()
    
    if global_pred > 0 and global_actual > global_pred:
        raw_factor = global_actual / global_pred
        factor = min(raw_factor, max_factor)
    else:
        factor = 1.0
        
    # Rolling evaluation by date
    rolling_report = []
    df_sorted = df.sort_values("date")
    dates = df_sorted["date"].unique()
    
    for d in dates:
        d_str = str(pd.to_datetime(d).date())
        # Up to this date
        sub_df = df_sorted[df_sorted["date"] <= d]
        
        matches_evaluated = len(sub_df)
        avg_act = sub_df["actual_goals"].mean()
        avg_pred = sub_df["predicted_goals"].mean()
        goal_gap = avg_act - avg_pred
        
        rolling_report.append({
            "date": d_str,
            "matches_evaluated": matches_evaluated,
            "top1_exact_hit_rate": round(sub_df["top1_exact"].mean(), 4),
            "top5_exact_hit_rate": round(sub_df["top5_exact"].mean(), 4),
            "1X2_hit_rate": round(sub_df["outcome_1x2"].mean(), 4),
            "average_actual_total_goals": round(avg_act, 3),
            "average_predicted_total_goals": round(avg_pred, 3),
            "goal_gap": round(goal_gap, 3)
        })
        
    report_output = {
        "as_of_date": as_of_date,
        "global_calibration_factor": round(factor, 3),
        "raw_calibration_factor": round(global_actual / global_pred, 3) if global_pred > 0 else 1.0,
        "average_actual_total_goals": round(global_actual, 3),
        "average_predicted_total_goals": round(global_pred, 3),
        "matches_evaluated": len(df),
        "rolling_metrics": rolling_report
    }
    
    out_path = OUTPUTS_DIR / "calibration_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2)
        
    return {
        "factor": round(factor, 3),
        "actual_goals_avg": round(global_actual, 3),
        "predicted_goals_avg": round(global_pred, 3),
        "reason": "Calibrated"
    }

if __name__ == "__main__":
    res = calculate_calibration_factor(as_of_date=None)
    print(res)
