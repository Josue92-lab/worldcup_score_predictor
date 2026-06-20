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
        return {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "Disabled in config"}
    
    max_factor = config.get("max_factor", 1.25)
    min_matches = config.get("min_matches_required", 12)
    rolling_window = config.get("rolling_window", 15)
    
    backtest_path = OUTPUTS_DIR / "backtest_report.json"
    if not backtest_path.exists():
        return {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "No backtest_report.json"}
    
    with open(backtest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    predictions = data.get("predictions", [])
    if not predictions:
        return {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "No predictions in backtest"}
        
    results_path = RAW_DIR / "international_results" / "results.csv"
    if not results_path.exists():
        return {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "No actual results.csv"}
        
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
        
    if len(wc2026_actuals) == 0:
        return {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "No actual matches yet"}
        
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
            pred_xg_a = float(p.get("expected_goals_team_a", 0.0))
            pred_xg_b = float(p.get("expected_goals_team_b", 0.0))
            pred_xg = pred_xg_a + pred_xg_b
            
            # Hit metrics
            if row["home_team_norm"] == p_team_a:
                actual_score_a = int(row["home_score"])
                actual_score_b = int(row["away_score"])
            else:
                actual_score_a = int(row["away_score"])
                actual_score_b = int(row["home_score"])
                
            actual_goals = actual_score_a + actual_score_b
            actual_scoreline = f"{actual_score_a}-{actual_score_b}"
            
            # Identify favorite vs underdog
            if pred_xg_a >= pred_xg_b:
                fav_xg, und_xg = pred_xg_a, pred_xg_b
                fav_actual, und_actual = actual_score_a, actual_score_b
            else:
                fav_xg, und_xg = pred_xg_b, pred_xg_a
                fav_actual, und_actual = actual_score_b, actual_score_a
            
            actual_outcome = "1" if actual_score_a > actual_score_b else "X" if actual_score_a == actual_score_b else "2"
            
            top5 = [s["scoreline"] for s in p["top_5_scorelines"]]
            win_a = float(p.get("team_a_win_probability", 0.0))
            draw = float(p.get("draw_probability", 0.0))
            win_b = float(p.get("team_b_win_probability", 0.0))
            
            # Use aggregate probabilities for 1X2 (same logic as evaluate)
            if win_a > draw and win_a > win_b:
                pred_outcome = "1"
            elif draw > win_a and draw > win_b:
                pred_outcome = "X"
            elif win_b > win_a and win_b > draw:
                pred_outcome = "2"
            else:
                if win_a == draw and win_a > win_b:
                    pred_outcome = "X"
                elif win_b == draw and win_b > win_a:
                    pred_outcome = "X"
                else:
                    pred_outcome = "X"
                
            is_top1 = actual_scoreline == top5[0] if top5 else False
            is_top5 = actual_scoreline in top5
            is_outcome = actual_outcome == pred_outcome
            
            matched_data.append({
                "date": str(p_date),
                "actual_goals": actual_goals,
                "predicted_goals": pred_xg,
                "fav_actual": fav_actual,
                "fav_xg": fav_xg,
                "und_actual": und_actual,
                "und_xg": und_xg,
                "top1_exact": is_top1,
                "top5_exact": is_top5,
                "outcome_1x2": is_outcome
            })
            
    df = pd.DataFrame(matched_data)
    if df.empty:
        return {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "No matched rows"}
        
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # 1. Rolling Window
    df_recent = df.tail(rolling_window)
    
    global_actual = df_recent["actual_goals"].mean()
    global_pred = df_recent["predicted_goals"].mean()
    
    fav_actual_sum = df_recent["fav_actual"].sum()
    fav_xg_sum = df_recent["fav_xg"].sum()
    und_actual_sum = df_recent["und_actual"].sum()
    und_xg_sum = df_recent["und_xg"].sum()
    
    raw_factor_fav = (fav_actual_sum / fav_xg_sum) if fav_xg_sum > 0 else 1.0
    raw_factor_und = (und_actual_sum / und_xg_sum) if und_xg_sum > 0 else 1.0
    
    # 3. Combinar Señales: Confianza basada en precisión 1X2
    hit_rate = df_recent["outcome_1x2"].mean()
    confidence_multiplier = 1.0 if hit_rate < 0.5 else 1.1

    if len(df_recent) < min_matches:
        # 4. Factor Suave Inicial
        smoothing_weight = len(df_recent) / min_matches
        
        # Smooth with half-step
        factor_fav = 1.0 + ((raw_factor_fav * confidence_multiplier - 1.0) * smoothing_weight * 0.5)
        factor_und = 1.0 + ((raw_factor_und * confidence_multiplier - 1.0) * smoothing_weight * 0.5)
        
        factor_fav = min(max(factor_fav, 0.90), 1.10)
        factor_und = min(max(factor_und, 0.90), 1.10)
        factor_global = (factor_fav + factor_und) / 2.0
        reason_msg = f"Smooth factor (only {len(df_recent)} matches)"
    else:
        # Half-step smoothing for strict bounds to prevent inversion of favorites
        factor_fav = 1.0 + (raw_factor_fav * confidence_multiplier - 1.0) * 0.5
        factor_und = 1.0 + (raw_factor_und * confidence_multiplier - 1.0) * 0.5
        
        factor_fav = min(max(factor_fav, 0.90), 1.10)
        factor_und = min(max(factor_und, 0.90), 1.10)
        factor_global = (factor_fav + factor_und) / 2.0
        reason_msg = "Rolling Window Calibrated"

    # Reporte de métricas usando df completo para historico
    rolling_report = []
    dates = df["date"].unique()
    
    for d in dates:
        d_str = str(pd.to_datetime(d).date())
        sub_df = df[df["date"] <= d]
        
        rolling_report.append({
            "date": d_str,
            "matches_evaluated": len(sub_df),
            "top1_exact_hit_rate": round(sub_df["top1_exact"].mean(), 4),
            "top5_exact_hit_rate": round(sub_df["top5_exact"].mean(), 4),
            "1X2_hit_rate": round(sub_df["outcome_1x2"].mean(), 4),
            "average_actual_total_goals": round(sub_df["actual_goals"].mean(), 3),
            "average_predicted_total_goals": round(sub_df["predicted_goals"].mean(), 3),
            "goal_gap": round(sub_df["actual_goals"].mean() - sub_df["predicted_goals"].mean(), 3)
        })
        
    report_output = {
        "as_of_date": as_of_date,
        "global_calibration_factor": round(factor_global, 3),
        "favorite_calibration_factor": round(factor_fav, 3),
        "underdog_calibration_factor": round(factor_und, 3),
        "matches_evaluated_in_window": len(df_recent),
        "average_actual_total_goals": round(global_actual, 3),
        "average_predicted_total_goals": round(global_pred, 3),
        "rolling_metrics": rolling_report
    }
    
    out_path = OUTPUTS_DIR / "calibration_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_output, f, indent=2)
        
    return {
        "factor": round(factor_global, 3),
        "factor_fav": round(factor_fav, 3),
        "factor_und": round(factor_und, 3),
        "actual_goals_avg": round(global_actual, 3),
        "predicted_goals_avg": round(global_pred, 3),
        "reason": reason_msg
    }

if __name__ == "__main__":
    res = calculate_calibration_factor(as_of_date=None)
    print(res)
