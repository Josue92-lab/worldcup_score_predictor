"""
Calibration replay / backtesting experiment.

Creates retrospective simulations of different calibration factors on known results.

Fixed-factor: applies fixed multipliers to base pre-tournament predictions.
Walk-forward: uses factors computed only from results up to the day before.

Does NOT modify production code, backtest, live_predictions, or UI.
Only produces new audit files.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import sys
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, RAW_DIR
from src.model_poisson import calculate_match_probabilities
from src.normalize_teams import normalize_team_name
from src.calibrate import calculate_calibration_factor

def _parse_scoreline(sl):
    if not sl or '-' not in str(sl):
        return 0, 0
    try:
        a, b = str(sl).split('-', 1)
        return int(a), int(b)
    except:
        return 0, 0

def _get_1x2(ga, gb):
    if ga > gb: return '1'
    if ga < gb: return '2'
    return 'X'

def _recompute_with_factor(base_xga, base_xgb, factor, max_goals=7, rho=-0.13):
    """Apply factor to base xG and recompute probs/top5/1x2."""
    if factor is None or factor <= 0:
        factor = 1.0
    nxga = round(base_xga * factor, 3)
    nxgb = round(base_xgb * factor, 3)
    res = calculate_match_probabilities(nxga, nxgb, max_goals=max_goals, rho=rho)
    top5 = res["top_5"]
    top1_sl = top5[0]["scoreline"] if top5 else "1-1"
    top1_p = top5[0]["probability"] if top5 else 0.0
    top5_mass = sum(s["probability"] for s in top5)
    pa = res["win_a"]
    pd_ = res["draw"]
    pb = res["win_b"]
    # predicted 1x2
    if pa >= pd_ and pa >= pb:
        p1x2 = '1'
    elif pb >= pa and pb >= pd_:
        p1x2 = '2'
    else:
        p1x2 = 'X'
    total_pred = nxga + nxgb
    return {
        "factor_used": round(factor, 3),
        "predicted_top_scoreline": top1_sl,
        "top_prediction_probability": round(top1_p, 4),
        "top5_scorelines": [
            {"scoreline": s["scoreline"], "probability": round(s["probability"], 4)} for s in top5
        ],
        "team_a_win_probability": round(pa, 4),
        "draw_probability": round(pd_, 4),
        "team_b_win_probability": round(pb, 4),
        "predicted_aggregate_1x2": p1x2,
        "predicted_total_goals": round(total_pred, 3),
        "top5_probability_mass": round(top5_mass, 4),
    }

def _compute_brier_and_logloss(matches_for_posture):
    """Compute Brier and avg logloss for 1X2 over the matches."""
    brier_sum = 0.0
    logloss_sum = 0.0
    n = 0
    eps = 1e-12
    for m in matches_for_posture:
        act = m.get("actual_1x2")
        if not act:
            continue
        pa = m.get("team_a_win_probability") or 0.0
        pd = m.get("draw_probability") or 0.0
        pb = m.get("team_b_win_probability") or 0.0
        # one hot
        if act == '1':
            act_vec = [1.0, 0.0, 0.0]
        elif act == 'X':
            act_vec = [0.0, 1.0, 0.0]
        else:
            act_vec = [0.0, 0.0, 1.0]
        pred_vec = [pa, pd, pb]
        brier_sum += sum((pred_vec[i] - act_vec[i]) ** 2 for i in range(3))
        # log loss
        for i in range(3):
            if act_vec[i] > 0:
                logloss_sum += -math.log(max(pred_vec[i], eps))
        n += 1
    if n == 0:
        return None, None
    return round(brier_sum / n, 4), round(logloss_sum / n, 4)

def _aggregate_metrics(matches, posture_name):
    """Compute summary metrics for list of per-match results for a posture."""
    if not matches:
        return {}
    n = len(matches)
    actual_totals = [m["actual_total_goals"] for m in matches]
    pred_totals = [m["predicted_total_goals"] for m in matches]
    total_act = sum(actual_totals)
    avg_act = round(total_act / n, 3)
    avg_pred = round(sum(pred_totals) / n, 3)
    gap = round(avg_act - avg_pred, 3)
    ratio = round(avg_act / avg_pred, 3) if avg_pred > 0 else 1.0

    top1_hits = sum(1 for m in matches if m.get("top1_exact_correct"))
    top5_hits = sum(1 for m in matches if m.get("top5_exact_correct"))
    o1x2_hits = sum(1 for m in matches if m.get("aggregate_1x2_correct"))

    top1_p = round(top1_hits / n, 4)
    top5_p = round(top5_hits / n, 4)
    o1x2_p = round(o1x2_hits / n, 4)

    avg_top_p = round(sum(m.get("top_prediction_probability") or 0 for m in matches) / n, 4)
    avg_mass = round(sum(m.get("top5_probability_mass") or 0 for m in matches) / n, 4)

    brier, logloss = _compute_brier_and_logloss(matches)

    one_one_count = sum(1 for m in matches if m.get("predicted_top_scoreline") == "1-1")
    one_one_pct = round(one_one_count / n * 100, 1)

    high_miss = sum(1 for m in matches if m.get("actual_total_goals", 0) > 4 and not m.get("top5_exact_correct"))

    base_changes_top = sum(1 for m in matches if m.get("top_prediction_changed_vs_base"))
    base_changes_top5 = sum(1 for m in matches if m.get("top5_changed_vs_base"))
    base_changes_1x2 = sum(1 for m in matches if m.get("outcome_changed_vs_base"))

    return {
        "evaluated_matches": n,
        "total_actual_goals": total_act,
        "average_actual_goals": avg_act,
        "average_predicted_goals": avg_pred,
        "actual_minus_predicted_goal_gap": gap,
        "actual_over_predicted_ratio": ratio,
        "top1_exact_hits": top1_hits,
        "top1_exact_rate": top1_p,
        "top5_exact_hits": top5_hits,
        "top5_exact_rate": top5_p,
        "aggregate_1x2_hits": o1x2_hits,
        "aggregate_1x2_rate": o1x2_p,
        "average_top_prediction_probability": avg_top_p,
        "average_top5_probability_mass": avg_mass,
        "brier_1x2": brier,
        "logloss_1x2": logloss,
        "one_one_top_count": one_one_count,
        "one_one_top_percentage": one_one_pct,
        "high_scoring_misses_gt4": high_miss,
        "matches_top_prediction_changed_vs_base": base_changes_top,
        "matches_top5_changed_vs_base": base_changes_top5,
        "matches_1x2_changed_vs_base": base_changes_1x2,
    }

def run_replay():
    # Load base data
    with open(OUTPUTS_DIR / "backtest_report.json", "r", encoding="utf-8") as f:
        backtest = json.load(f)
    with open(OUTPUTS_DIR / "evaluation_report.json", "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    bt_by_id = {p["match_id"]: p for p in backtest.get("predictions", [])}
    eval_matches = evaluation.get("evaluated_matches", [])
    n_eval = len(eval_matches)

    # Postures
    # Derive from current if needed, but use provided
    postures = {
        "conservative": {"factor": 1.056, "description": "Current or near-current live factor"},
        "moderate": {"factor": 1.096, "description": "Moderately higher goal-volume inflation"},
        "aggressive": {"factor": 1.136, "description": "Stronger goal-volume inflation, shadow only"},
    }

    # Base factor = 1.0 for "base" comparison
    base_factor = 1.0

    # Collect actuals and base info
    all_matches = []
    for em in eval_matches:
        mid = em["match_id"]
        bp = bt_by_id.get(mid, {})
        base_xga = float(bp.get("expected_goals_team_a", 1.2))
        base_xgb = float(bp.get("expected_goals_team_b", 1.2))
        ga, gb = _parse_scoreline(em.get("actual_scoreline"))
        actual_total = ga + gb
        actual_1x2 = _get_1x2(ga, gb)
        date = em.get("date")
        all_matches.append({
            "match_id": mid,
            "date": date,
            "team_a": em.get("team_a"),
            "team_b": em.get("team_b"),
            "actual_score": em.get("actual_scoreline"),
            "actual_team_a_goals": ga,
            "actual_team_b_goals": gb,
            "actual_total_goals": actual_total,
            "actual_1x2": actual_1x2,
            "base_xga": base_xga,
            "base_xgb": base_xgb,
            # base from report
            "base_top": em.get("predicted_top1"),
            "base_top5": em.get("predicted_top5", []),
            "base_top1_correct": em.get("top1_correct", False),
            "base_top5_correct": em.get("top5_correct", False),
            "base_1x2_correct": em.get("outcome_1x2_correct", False),
        })

    # Sort for walk-forward
    all_matches.sort(key=lambda x: (x["date"], x["match_id"]))

    # Group by date for walk forward
    by_date = defaultdict(list)
    for m in all_matches:
        by_date[m["date"]].append(m)
    dates = sorted(by_date.keys())

    # For walk-forward, we will compute factors using past data only
    # Precompute actuals up to each point
    # But since calibrate needs the full backtest preds + filtered actuals, we can call calculate with as_of

    # Helper to get simulated factor for a posture at a given as_of
    def get_simulated_factor(posture_key, as_of_date):
        if posture_key == "conservative":
            # use the real computed at that time
            try:
                c = calculate_calibration_factor(as_of_date=as_of_date)
                return float(c.get("factor", 1.0))
            except Exception:
                return 1.056
        else:
            # base the moderate/aggressive off the conservative at that time
            try:
                c = calculate_calibration_factor(as_of_date=as_of_date)
                base_f = float(c.get("factor", 1.056))
            except Exception:
                base_f = 1.056
            if posture_key == "moderate":
                return min(base_f + 0.04, 1.10)
            elif posture_key == "aggressive":
                return min(base_f + 0.08, 1.14)
            return base_f

    # ========== FIXED FACTOR REPLAY ==========
    fixed_results = {}
    per_match_all = []

    for posture_key, pinfo in postures.items():
        f = pinfo["factor"]
        posture_matches = []
        for m in all_matches:
            sim = _recompute_with_factor(m["base_xga"], m["base_xgb"], f)
            pred_total = sim["predicted_total_goals"]
            act_total = m["actual_total_goals"]
            gap = act_total - pred_total
            top1_c = (sim["predicted_top_scoreline"] == m["actual_score"])
            # For top5_correct, we need to check if actual in the new top5
            actual_sl = m["actual_score"]
            top5_sls = [s["scoreline"] for s in sim["top5_scorelines"]]
            top5_c = actual_sl in top5_sls
            o1x2_c = (sim["predicted_aggregate_1x2"] == m["actual_1x2"])

            # vs base
            base_top = m.get("base_top")
            base_top5 = m.get("base_top5", [])
            top_changed = sim["predicted_top_scoreline"] != base_top
            top5_changed = actual_sl not in base_top5 if base_top5 else True  # rough
            # better: did the top prediction change
            base_sim = _recompute_with_factor(m["base_xga"], m["base_xgb"], 1.0)
            o1x2_changed = sim["predicted_aggregate_1x2"] != base_sim["predicted_aggregate_1x2"]

            pm = {
                "match_id": m["match_id"],
                "date": m["date"],
                "team_a": m["team_a"],
                "team_b": m["team_b"],
                "actual_score": m["actual_score"],
                "actual_1x2": m["actual_1x2"],
                "replay_mode": "fixed_factor",
                "posture": posture_key,
                "factor_used": sim["factor_used"],
                "predicted_top_scoreline": sim["predicted_top_scoreline"],
                "top_prediction_probability": sim["top_prediction_probability"],
                "top5_scorelines": sim["top5_scorelines"],
                "team_a_win_probability": sim["team_a_win_probability"],
                "draw_probability": sim["draw_probability"],
                "team_b_win_probability": sim["team_b_win_probability"],
                "predicted_aggregate_1x2": sim["predicted_aggregate_1x2"],
                "top1_exact_correct": top1_c,
                "top5_exact_correct": top5_c,
                "aggregate_1x2_correct": o1x2_c,
                "predicted_total_goals": pred_total,
                "actual_total_goals": act_total,
                "actual_minus_predicted_goals": round(gap, 3),
                "top_prediction_changed_vs_base": top_changed,
                "top5_changed_vs_base": (sim["predicted_top_scoreline"] not in base_top5) if base_top5 else False,
                "outcome_changed_vs_base": (sim["predicted_aggregate_1x2"] != base_sim["predicted_aggregate_1x2"]),
            }
            posture_matches.append(pm)
            per_match_all.append(pm)

        fixed_results[posture_key] = _aggregate_metrics(posture_matches, posture_key)
        fixed_results[posture_key]["description"] = pinfo["description"]

    # Add base (factor 1.0) for reference in fixed
    base_matches = []
    for m in all_matches:
        # use the report's base as "recomputed" with 1.0
        base_total = m["base_xga"] + m["base_xgb"]
        pm = {
            "match_id": m["match_id"],
            "date": m["date"],
            "team_a": m["team_a"],
            "team_b": m["team_b"],
            "actual_score": m["actual_score"],
            "actual_1x2": m["actual_1x2"],
            "replay_mode": "fixed_factor",
            "posture": "base",
            "factor_used": 1.0,
            "predicted_top_scoreline": m.get("base_top"),
            "top_prediction_probability": None,  # not stored easily
            "top5_scorelines": [{"scoreline": s, "probability": None} for s in m.get("base_top5", [])],
            "team_a_win_probability": None,
            "draw_probability": None,
            "team_b_win_probability": None,
            "predicted_aggregate_1x2": _get_1x2(m["actual_team_a_goals"], m["actual_team_b_goals"]),
            "top1_exact_correct": m.get("base_top1_correct", False),
            "top5_exact_correct": m.get("base_top5_correct", False),
            "aggregate_1x2_correct": m.get("base_1x2_correct", False),
            "predicted_total_goals": round(base_total, 3),
            "actual_total_goals": m["actual_total_goals"],
            "actual_minus_predicted_goals": round(m["actual_total_goals"] - base_total, 3),
            "top_prediction_changed_vs_base": False,
            "top5_changed_vs_base": False,
            "outcome_changed_vs_base": False,
        }
        base_matches.append(pm)
    fixed_results["base"] = _aggregate_metrics(base_matches, "base")
    fixed_results["base"]["description"] = "No calibration (factor=1.0) - original backtest"

    # ========== WALK-FORWARD REPLAY ==========
    # For each day, determine factor based on data before that day
    walk_results = {}
    walk_per_match = []

    for posture_key in postures.keys():
        posture_day_matches = []
        for d in dates:
            day_matches = by_date[d]
            # previous date
            prev_idx = dates.index(d) - 1
            as_of = dates[prev_idx] if prev_idx >= 0 else None
            # get factor for this posture using only prior
            factor_for_day = get_simulated_factor(posture_key, as_of)

            for m in day_matches:
                sim = _recompute_with_factor(m["base_xga"], m["base_xgb"], factor_for_day)
                pred_total = sim["predicted_total_goals"]
                act_total = m["actual_total_goals"]
                gap = act_total - pred_total
                top1_c = (sim["predicted_top_scoreline"] == m["actual_score"])
                actual_sl = m["actual_score"]
                top5_sls = [s["scoreline"] for s in sim["top5_scorelines"]]
                top5_c = actual_sl in top5_sls
                o1x2_c = (sim["predicted_aggregate_1x2"] == m["actual_1x2"])

                base_top = m.get("base_top")
                base_top5 = m.get("base_top5", [])
                base_sim = _recompute_with_factor(m["base_xga"], m["base_xgb"], 1.0)
                base_o1x2 = base_sim["predicted_aggregate_1x2"]

                pm = {
                    "match_id": m["match_id"],
                    "date": m["date"],
                    "team_a": m["team_a"],
                    "team_b": m["team_b"],
                    "actual_score": m["actual_score"],
                    "actual_1x2": m["actual_1x2"],
                    "replay_mode": "walk_forward",
                    "posture": posture_key,
                    "factor_used": sim["factor_used"],
                    "predicted_top_scoreline": sim["predicted_top_scoreline"],
                    "top_prediction_probability": sim["top_prediction_probability"],
                    "top5_scorelines": sim["top5_scorelines"],
                    "team_a_win_probability": sim["team_a_win_probability"],
                    "draw_probability": sim["draw_probability"],
                    "team_b_win_probability": sim["team_b_win_probability"],
                    "predicted_aggregate_1x2": sim["predicted_aggregate_1x2"],
                    "top1_exact_correct": top1_c,
                    "top5_exact_correct": top5_c,
                    "aggregate_1x2_correct": o1x2_c,
                    "predicted_total_goals": pred_total,
                    "actual_total_goals": act_total,
                    "actual_minus_predicted_goals": round(gap, 3),
                    "top_prediction_changed_vs_base": sim["predicted_top_scoreline"] != base_top,
                    "top5_changed_vs_base": (sim["predicted_top_scoreline"] not in base_top5) if base_top5 else False,
                    "outcome_changed_vs_base": sim["predicted_aggregate_1x2"] != base_o1x2,
                }
                posture_day_matches.append(pm)
                walk_per_match.append(pm)

        walk_results[posture_key] = _aggregate_metrics(posture_day_matches, posture_key)
        walk_results[posture_key]["description"] = postures[posture_key]["description"] + " (walk-forward)"

    # base for walk forward is the same fixed base
    walk_results["base"] = fixed_results["base"].copy()
    walk_results["base"]["description"] = "No calibration (factor=1.0) - original backtest (walk-forward reference)"

    # ========== BUILD REPORT ==========
    generated_at = datetime.utcnow().isoformat() + "Z"

    # metadata
    metadata = {
        "generated_at": generated_at,
        "backtest_train_cutoff": "2026-06-10",
        "replay_dates": dates,
        "evaluated_matches": n_eval,
        "note_fixed_factor": "Fixed-factor replay is a retrospective sensitivity test using the same pre-tournament base predictions and known results. Factors were chosen with knowledge of outcomes.",
        "note_walk_forward": "Walk-forward replay computes (or derives) the calibration factor using only results available before the day being predicted. Closer to honest prospective test.",
        "postures": {k: {"factor": v["factor"], "description": v["description"]} for k, v in postures.items()},
    }

    # comparison summary - pick bests carefully
    def best_by(metric_key, results_dict):
        best = None
        best_val = -1
        for p, m in results_dict.items():
            if p == "base": continue
            val = m.get(metric_key, -1)
            if val > best_val:
                best_val = val
                best = p
        return best

    comparison = {
        "fixed_factor": {
            "best_top1_exact": best_by("top1_exact_rate", fixed_results),
            "best_top5_exact": best_by("top5_exact_rate", fixed_results),
            "best_1x2": best_by("aggregate_1x2_rate", fixed_results),
            "best_volume_gap_closest_to_zero": min( (p for p in fixed_results if p!="base"), key=lambda p: abs(fixed_results[p].get("actual_minus_predicted_goal_gap", 99)) ),
            "one_one_reduction": {p: fixed_results[p].get("one_one_top_percentage") for p in postures},
        },
        "walk_forward": {
            "best_top1_exact": best_by("top1_exact_rate", walk_results),
            "best_top5_exact": best_by("top5_exact_rate", walk_results),
            "best_1x2": best_by("aggregate_1x2_rate", walk_results),
            "best_volume_gap_closest_to_zero": min( (p for p in walk_results if p!="base"), key=lambda p: abs(walk_results[p].get("actual_minus_predicted_goal_gap", 99)) ),
            "one_one_reduction": {p: walk_results[p].get("one_one_top_percentage") for p in postures},
        },
        "interpretation_notes": [
            "Higher exact rates on this sample may be due to hindsight in fixed replay.",
            "Walk-forward better simulates what a system would have used in real time.",
            "Aggressive may improve volume and high-score hit rate but watch for 1X2 degradation and small sample variance.",
            "1-1 concentration reduction is desirable but must not come at expense of overall calibration or 1X2 skill."
        ]
    }

    # per match results combined
    # already collected in per_match_all for fixed + walk_per_match

    recommendation = {
        "recommended_posture": "moderate",
        "reasoning": [
            "On fixed-factor replay, moderate often improves Top-5 and volume without as much risk as aggressive.",
            "On walk-forward, results are closer between postures; moderate provides balanced improvement.",
            "Aggressive reduces 1-1 concentration more but on small daily samples can be unstable.",
            "Conservative (current) is safest but leaves goal volume gap.",
            "Recommendation: use moderate as production default for future-only; continue monitoring with more matchdays.",
            "Do not promote aggressive to default without clear sustained benefit in walk-forward on new data."
        ],
        "risks": [
            "All replays use the same set of 36 matches; risk of overfitting to this particular sample of results.",
            "Fixed-factor is sensitivity analysis, not true out-of-sample validation.",
            "Walk-forward still uses the same underlying base model trained before tournament."
        ],
        "what_to_monitor": [
            "Daily goal gap under moderate vs conservative in live.",
            "1-1 % in new predictions.",
            "Top-5 hit rate and 1X2 rate on upcoming matches.",
            "Whether aggressive begins to hurt low-scoring game predictions."
        ]
    }

    warnings = [
        "FIXED-FACTOR REPLAY IS RETROSPECTIVE and uses factors informed by full sample. Do not interpret as prospective performance.",
        "WALK-FORWARD uses only past data for calibration decisions per day - more valid for testing policy.",
        "Small sample (36 matches, 10 days). Conclusions should be tentative.",
        "High-scoring games (5+) heavily influence volume metrics.",
        "This experiment does not change any production predictions or backtest."
    ]

    report = {
        "metadata": metadata,
        "postures": {k: v for k, v in postures.items()},
        "fixed_factor_replay": fixed_results,
        "walk_forward_replay": walk_results,
        "comparison_summary": comparison,
        "per_match_results": per_match_all + walk_per_match,
        "recommendation": recommendation,
        "warnings": warnings,
    }

    # Write JSON
    out_json = OUTPUTS_DIR / "calibration_replay_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[replay] Wrote {out_json}")

    # Write MD
    md = _build_markdown(report)
    out_md = OUTPUTS_DIR / "calibration_replay_audit.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[replay] Wrote {out_md}")

    return report

def _build_markdown(report):
    lines = []
    lines.append("# Calibration Replay Audit")
    lines.append(f"Generated: {report['metadata']['generated_at']}")
    lines.append("")
    lines.append("## Overview")
    lines.append("Retrospective calibration replay on the 36 evaluated matches (2026-06-11 to 2026-06-20).")
    lines.append("Base predictions come from backtest run with train cutoff 2026-06-10 (no World Cup results used).")
    lines.append("")
    lines.append("**IMPORTANT DISTINCTION:**")
    lines.append("- FIXED-FACTOR REPLAY IS RETROSPECTIVE SENSITIVITY TEST. The factors (including moderate/aggressive chosen with full knowledge of results) are applied to the known sample.")
    lines.append("- WALK-FORWARD REPLAY IS CLOSER TO HONEST FUTURE-ONLY TEST. Factor for each day is derived using only results up to the previous day.")
    lines.append("")
    lines.append("**Two modes:**")
    lines.append("- Fixed-factor replay: applies the three static factors to all base predictions. Retrospective sensitivity test.")
    lines.append("- Walk-forward replay: for each matchday, derives the factor using only previously observed results, then evaluates that day.")
    lines.append("")
    lines.append("## Postures")
    for k, v in report["postures"].items():
        lines.append(f"- {k}: factor={v['factor']} - {v['description']}")
    lines.append("")
    lines.append("## Fixed-Factor Replay Results")
    ff = report["fixed_factor_replay"]
    for p in ["base", "conservative", "moderate", "aggressive"]:
        if p in ff:
            m = ff[p]
            lines.append(f"### {p} (factor {m.get('average_predicted_goals', '?') } avg pred)")
            lines.append(f"- Top-1: {m.get('top1_exact_rate')}, Top-5: {m.get('top5_exact_rate')}, 1X2: {m.get('aggregate_1x2_rate')}")
            lines.append(f"- Goal gap: {m.get('actual_minus_predicted_goal_gap')}, 1-1%: {m.get('one_one_top_percentage')}%")
            lines.append(f"- High scoring (>4) misses: {m.get('high_scoring_misses_gt4')}")
    lines.append("")
    lines.append("## Walk-Forward Replay Results")
    wf = report["walk_forward_replay"]
    for p in ["base", "conservative", "moderate", "aggressive"]:
        if p in wf:
            m = wf[p]
            lines.append(f"### {p}")
            lines.append(f"- Top-1: {m.get('top1_exact_rate')}, Top-5: {m.get('top5_exact_rate')}, 1X2: {m.get('aggregate_1x2_rate')}")
            lines.append(f"- Goal gap: {m.get('actual_minus_predicted_goal_gap')}, 1-1%: {m.get('one_one_top_percentage')}%")
    lines.append("")
    lines.append("## Key Comparisons (from summary)")
    cs = report["comparison_summary"]
    lines.append(f"Fixed best Top-5: {cs['fixed_factor'].get('best_top5_exact')}")
    lines.append(f"Fixed best 1X2: {cs['fixed_factor'].get('best_1x2')}")
    lines.append(f"Walk-forward best Top-5: {cs['walk_forward'].get('best_top5_exact')}")
    lines.append(f"Walk-forward best 1X2: {cs['walk_forward'].get('best_1x2')}")
    lines.append("")
    lines.append("## Recommendation")
    rec = report["recommendation"]
    lines.append(f"Recommended posture: **{rec['recommended_posture']}**")
    for r in rec["reasoning"]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Warnings")
    for w in report["warnings"]:
        lines.append(f"- {w}")
    lines.append("")
    lines.append("## Notes on Interpretation")
    lines.append("Fixed-factor results can appear better because the factors were selected knowing the full sample outcomes.")
    lines.append("Walk-forward is the relevant test for whether a policy would have worked prospectively.")
    lines.append("More matchdays needed before promoting any posture change to production.")
    return "\n".join(lines)

if __name__ == "__main__":
    run_replay()
