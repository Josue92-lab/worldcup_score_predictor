import json
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, RAW_DIR
from src.normalize_teams import normalize_team_name
from src.outcomes import get_predicted_1x2_outcome, get_actual_1x2_outcome, is_1x2_hit


def evaluate_predictions(as_of_date=None):
    backtest_path = OUTPUTS_DIR / "backtest_report.json"
    if not backtest_path.exists():
        print(f"Error: {backtest_path} not found. Run backtest first.")
        return

    with open(backtest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        print("No predictions found to evaluate.")
        return

    results_path = RAW_DIR / "international_results" / "results.csv"
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run download-data first.")
        return

    actuals = pd.read_csv(results_path)
    actuals["date"] = pd.to_datetime(actuals["date"], errors="coerce")
    
    max_date = str(actuals["date"].dt.date.max())
    
    actuals["home_team_norm"] = actuals["home_team"].apply(normalize_team_name)
    actuals["away_team_norm"] = actuals["away_team"].apply(normalize_team_name)
    
    # Filter World Cup 2026 matches
    wc2026_actuals = actuals[
        (actuals["date"] >= pd.to_datetime("2026-06-11")) &
        (actuals["tournament"].str.contains("World Cup", case=False, na=False))
    ].copy()
    
    if as_of_date:
        wc2026_actuals = wc2026_actuals[wc2026_actuals["date"] <= pd.to_datetime(as_of_date)]
        
    world_cup_2026_rows_found = len(wc2026_actuals)
    
    if world_cup_2026_rows_found == 0:
        report = {
            "martj42_results_max_date": max_date,
            "world_cup_2026_rows_found": 0,
            "actual_results_status": "not_available"
        }
        eval_report_path = OUTPUTS_DIR / "evaluation_report.json"
        with open(eval_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("No actual World Cup 2026 results exist in martj42.")
        return

    matches_evaluated = 0
    exact_score_top1_hits = 0
    exact_score_top5_hits = 0
    outcome_1x2_hits = 0
    
    evaluated_matches = []

    top1_scoreline_distribution = {}
    total_xg_a = 0.0
    total_xg_b = 0.0
    
    total_actual_goals = 0.0
    total_predicted_goals = 0.0
    matches_where_actual_total_goals_gt_4 = 0
    high_score_miss_count = 0

    for p in predictions:
        p_date = pd.to_datetime(p["date"]).date()
        p_team_a = p["team_a"]
        p_team_b = p["team_b"]
        
        total_xg_a += float(p.get("expected_goals_team_a", 0.0))
        total_xg_b += float(p.get("expected_goals_team_b", 0.0))
        
        # Find match in actuals
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

            if row["home_team_norm"] == p_team_a:
                actual_score_a = int(row["home_score"])
                actual_score_b = int(row["away_score"])
            else:
                actual_score_a = int(row["away_score"])
                actual_score_b = int(row["home_score"])

            actual_scoreline = f"{actual_score_a}-{actual_score_b}"
            
            actual_outcome = get_actual_1x2_outcome(actual_score_a, actual_score_b)
                
            top5 = [s["scoreline"] for s in p["top_5_scorelines"]]
            
            # Count top-1 only for actually evaluated matches (prevents count > matches_evaluated)
            if top5:
                top_score = top5[0]
                top1_scoreline_distribution[top_score] = top1_scoreline_distribution.get(top_score, 0) + 1
            
            win_a = float(p.get("team_a_win_probability", 0.0))
            draw = float(p.get("draw_probability", 0.0))
            win_b = float(p.get("team_b_win_probability", 0.0))
            
            pred_outcome = get_predicted_1x2_outcome(win_a, draw, win_b)
            
            is_top1 = actual_scoreline == top5[0] if top5 else False
            is_top5 = actual_scoreline in top5
            is_outcome = is_1x2_hit(win_a, draw, win_b, actual_score_a, actual_score_b)

            if is_top1:
                exact_score_top1_hits += 1
            if is_top5:
                exact_score_top5_hits += 1
            if is_outcome:
                outcome_1x2_hits += 1
                
            actual_goals = actual_score_a + actual_score_b
            pred_goals = float(p.get("expected_goals_team_a", 0.0)) + float(p.get("expected_goals_team_b", 0.0))
            
            total_actual_goals += actual_goals
            total_predicted_goals += pred_goals
            
            if actual_goals > 4:
                matches_where_actual_total_goals_gt_4 += 1
                if not is_top5:
                    high_score_miss_count += 1
                
            matches_evaluated += 1
            
            evaluated_matches.append({
                "match_id": p["match_id"],
                "date": str(p_date),
                "team_a": p_team_a,
                "team_b": p_team_b,
                "actual_scoreline": actual_scoreline,
                "predicted_top1": top5[0] if top5 else None,
                "predicted_top5": top5,
                "top1_correct": is_top1,
                "top5_correct": is_top5,
                "outcome_1x2_correct": is_outcome
            })

    top1_scoreline_distribution = dict(sorted(top1_scoreline_distribution.items(), key=lambda item: item[1], reverse=True))
    avg_xg_a = total_xg_a / len(predictions) if predictions else 0.0
    avg_xg_b = total_xg_b / len(predictions) if predictions else 0.0
    
    avg_actual_goals = total_actual_goals / matches_evaluated if matches_evaluated > 0 else 0.0
    avg_predicted_goals = total_predicted_goals / matches_evaluated if matches_evaluated > 0 else 0.0
    
    metadata = data.get("metadata", {})
    calibration_factor = metadata.get("calibration_factor", 1.0)

    report = {
        "martj42_results_max_date": max_date,
        "world_cup_2026_rows_found": world_cup_2026_rows_found,
        "actual_results_status": "available" if matches_evaluated > 0 else "needs_review",
        "matches_evaluated": matches_evaluated,
        "exact_score_top1_hits": exact_score_top1_hits,
        "exact_score_top1_rate": round(exact_score_top1_hits / matches_evaluated, 4) if matches_evaluated > 0 else 0.0,
        "exact_score_top5_hits": exact_score_top5_hits,
        "exact_score_top5_rate": round(exact_score_top5_hits / matches_evaluated, 4) if matches_evaluated > 0 else 0.0,
        "outcome_1x2_hits": outcome_1x2_hits,
        "outcome_1x2_rate": round(outcome_1x2_hits / matches_evaluated, 4) if matches_evaluated > 0 else 0.0,
        "diagnostics": {
            "top1_scoreline_distribution": top1_scoreline_distribution,
            "average_expected_goals_team_a": round(avg_xg_a, 3),
            "average_expected_goals_team_b": round(avg_xg_b, 3),
            "predicted_total_goals_avg": round(avg_predicted_goals, 3),
            "actual_total_goals_avg": round(avg_actual_goals, 3),
            "high_score_miss_count": high_score_miss_count,
            "matches_where_actual_total_goals_gt_4": matches_where_actual_total_goals_gt_4,
            "calibration_factor": calibration_factor,
            "dixon_coles_enabled": True,
            "rho": -0.13
        },
        "evaluated_matches": evaluated_matches
    }

    eval_report_path = OUTPUTS_DIR / "evaluation_report.json"
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[evaluate] Evaluated {matches_evaluated} matches against actuals.")
    if matches_evaluated > 0:
        print(f"[evaluate] Top 1 Accuracy: {report['exact_score_top1_rate']:.1%}")
        print(f"[evaluate] Top 5 Accuracy: {report['exact_score_top5_rate']:.1%}")
    print(f"[evaluate] Saved report -> {eval_report_path}")

if __name__ == "__main__":
    evaluate_predictions()
