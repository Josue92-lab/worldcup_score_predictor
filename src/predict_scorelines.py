"""
Predict top-5 scorelines for each fixture using the Poisson model
and real feature data where available.
"""

import pandas as pd
import json
from pathlib import Path
import sys
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIXTURE_PATH, PROCESSED_DIR, PREDICTIONS_CSV_PATH, PREDICTIONS_JSON_PATH, RAW_DIR
from src.model_poisson import predict_xg, calculate_match_probabilities, fit_poisson_model
from src.normalize_teams import normalize_team_name
from src.build_features import load_model_config
from src.audit_data import is_knockout_placeholder
from src.calibrate import calculate_calibration_factor
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


def predict_scorelines(mode="live", as_of_date=None, train_cutoff=None, calibration_posture=None):
    if not FIXTURE_PATH.exists():
        print(f"Error: Fixture file {FIXTURE_PATH} not found.")
        return

    fixtures = pd.read_csv(FIXTURE_PATH)
    config = load_model_config()
    max_goals = config.get("model", {}).get("max_goals", 7)

    calibration_info = {"factor": 1.0, "factor_fav": 1.0, "factor_und": 1.0, "reason": "Not live mode"}
    applied_posture = None
    if mode == "live":
        base_calibration_info = calculate_calibration_factor(as_of_date)
        base_fav = base_calibration_info.get("factor_fav", 1.0)
        base_und = base_calibration_info.get("factor_und", 1.0)
        base_global = round((base_fav + base_und) / 2.0, 3) if (base_fav or base_und) else 1.0

        posture = calibration_posture or "moderate"
        applied_posture = posture

        if posture == "current":
            calib_factor_fav = base_fav
            calib_factor_und = base_und
            calib_reason = base_calibration_info.get("reason", "Rolling Window Calibrated")
        else:
            if posture == "moderate":
                target_global = min(base_global + 0.04, 1.10)
            elif posture == "aggressive":
                target_global = min(base_global + 0.08, 1.14)
            else:
                target_global = base_global
            scale = target_global / base_global if base_global > 0 else 1.0
            calib_factor_fav = round(base_fav * scale, 3)
            calib_factor_und = round(base_und * scale, 3)
            calib_reason = f"Challenger {posture} posture derived from base {base_global} (target {target_global})"

        calibration_info = {
            "factor": round((calib_factor_fav + calib_factor_und) / 2.0, 3),
            "factor_fav": calib_factor_fav,
            "factor_und": calib_factor_und,
            "reason": calib_reason
        }
        print(f"[predict] Live calibration posture={posture} - Fav: {calib_factor_fav:.3f}, Und: {calib_factor_und:.3f} ({calib_reason})")
    
    calib_factor_fav = calibration_info.get("factor_fav", 1.0)
    calib_factor_und = calibration_info.get("factor_und", 1.0)
    if mode == "live" and (calib_factor_fav != 1.0 or calib_factor_und != 1.0):
        print(f"[predict] Will apply posture multipliers only to future matches.")

    # Identify future matches so calibration applies only to them (past matches stay on base model)
    played_keys = set()
    if mode == "live":
        results_path = RAW_DIR / "international_results" / "results.csv"
        if results_path.exists():
            actuals = pd.read_csv(results_path)
            actuals["date"] = pd.to_datetime(actuals["date"], errors="coerce")
            wc = actuals[
                (actuals["date"] >= pd.to_datetime("2026-06-11")) &
                (actuals["tournament"].str.contains("World Cup", case=False, na=False)) &
                actuals["home_score"].notna() &
                actuals["away_score"].notna()
            ].copy()
            if not wc.empty:
                wc["h_norm"] = wc["home_team"].apply(normalize_team_name)
                wc["a_norm"] = wc["away_team"].apply(normalize_team_name)
                for _, r in wc.iterrows():
                    d = str(r["date"].date())
                    played_keys.add((d, r["h_norm"], r["a_norm"]))
                    played_keys.add((d, r["a_norm"], r["h_norm"]))

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
        
        # Apply calibration multipliers ONLY to future (unplayed) matches
        p_date = str(row.get("date", ""))
        is_played = (p_date, team_a, team_b) in played_keys or (p_date, team_b, team_a) in played_keys
        apply_calib = (mode == "live") and (not is_played) and (calib_factor_fav != 1.0 or calib_factor_und != 1.0)
        if apply_calib:
            if xg_a >= xg_b:
                xg_a = xg_a * calib_factor_fav
                xg_b = xg_b * calib_factor_und
            else:
                xg_b = xg_b * calib_factor_fav
                xg_a = xg_a * calib_factor_und
        
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
                "team_a_sqi": sq_a.get("squad_quality_index", 0),
                "team_b_sqi": sq_b.get("squad_quality_index", 0),
                "team_a_club_form": sq_a.get("recent_club_goals_per_app", 0),
                "team_b_club_form": sq_b.get("recent_club_goals_per_app", 0),
                "sqi_diff": sq_a.get("squad_quality_index", 0) - sq_b.get("squad_quality_index", 0),
                "club_form_diff": sq_a.get("recent_club_goals_per_app", 0) - sq_b.get("recent_club_goals_per_app", 0),
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
    if mode == "backtest":
        out_path = PREDICTIONS_JSON_PATH.parent / "backtest_report.json"
    elif mode == "live":
        out_path = PREDICTIONS_JSON_PATH.parent / "live_predictions.json"
    else:
        out_path = PREDICTIONS_JSON_PATH
    
    meta = {
        "mode": mode,
        "as_of_date": as_of_date,
        "train_cutoff": train_cutoff,
        "calibration_factor_fav": calib_factor_fav,
        "calibration_factor_und": calib_factor_und,
        "calibration_factor": round((calib_factor_fav + calib_factor_und) / 2.0, 3),
        "calibrated": bool(calib_factor_fav != 1.0 or calib_factor_und != 1.0),
        "calibration_reason": calibration_info.get("reason", "")
    }
    if mode == "live" and applied_posture:
        meta["calibration_posture"] = applied_posture
        meta["calibration_source"] = "performance audit / live goal-volume calibration"
        meta["future_only"] = True
        meta["not_for_backtest"] = True

    output_obj = {
        "metadata": meta,
        "predictions": predictions_json
    }

    # Determine output paths (challenger variants get separate files)
    if mode == "backtest":
        out_path = PREDICTIONS_JSON_PATH.parent / "backtest_report.json"
    elif mode == "live":
        posture = applied_posture or "moderate"
        out_path = PREDICTIONS_JSON_PATH.parent / f"live_predictions_{posture}.json"
    else:
        out_path = PREDICTIONS_JSON_PATH
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)
    print(f"[predict] Saved {len(predictions_json)} predictions -> {out_path}")

    # For moderate (production posture), also update the default live_predictions.json
    if mode == "live" and (applied_posture or "moderate") == "moderate":
        default_path = PREDICTIONS_JSON_PATH.parent / "live_predictions.json"
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, indent=2)
        print(f"[predict] Also saved default live_predictions.json using moderate posture")
    
    # Update PREDICTIONS_CSV_PATH based on mode (only for backtest/default live to avoid clutter)
    if mode == "backtest":
        out_csv_path = PREDICTIONS_CSV_PATH.parent / "backtest_report.csv"
    elif mode == "live" and (applied_posture or "moderate") == "moderate":
        out_csv_path = PREDICTIONS_CSV_PATH.parent / "live_predictions.csv"
    else:
        out_csv_path = None  # skip csv for challenger variants

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

    if out_csv_path:
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

    mv_a = sq_a.get("squad_quality_index", sq_a.get("squad_total_market_value", 0))
    mv_b = sq_b.get("squad_quality_index", sq_b.get("squad_total_market_value", 0))
    if mv_a > 0 and mv_b > 0:
        ratio = mv_a / mv_b if mv_b > 0 else 0
        if ratio > 2:
            factors.append(f"{team_a} squad quality is {ratio:.1f}x {team_b}")
        elif ratio < 0.5:
            factors.append(f"{team_b} squad quality is {1/ratio:.1f}x {team_a}")

    if not factors:
        factors.append("Baseline Poisson model with default strengths")

    return factors


def generate_calibration_challenger_report():
    """Generate calibration_challenger_report.json and .md from the three posture files.
    Assumes they have been created by predict with different postures.
    Only looks at future matches for diff analysis.
    """
    from datetime import datetime
    import copy

    out_dir = PREDICTIONS_JSON_PATH.parent
    postures = ["current", "moderate", "aggressive"]
    files = {p: out_dir / f"live_predictions_{p}.json" for p in postures}

    reports = {}
    for p, fp in files.items():
        if not fp.exists():
            print(f"[challenger] Warning: {fp} not found, skipping")
            reports[p] = {"metadata": {}, "predictions": []}
        else:
            with open(fp, "r", encoding="utf-8") as f:
                reports[p] = json.load(f)

    # Use moderate as reference for counts (same total)
    ref = reports.get("moderate", reports["current"])
    total_preds = len(ref.get("predictions", []))

    # Determine future matches using eval report if available, else all after a cutoff
    played_ids = set()
    eval_path = out_dir / "evaluation_report.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            ev = json.load(f)
        played_ids = {m["match_id"] for m in ev.get("evaluated_matches", [])}

    future_matches = []
    for pred in ref.get("predictions", []):
        if pred["match_id"] not in played_ids:
            future_matches.append(pred)

    n_future = len(future_matches)
    print(f"[challenger] Found {n_future} future matches for analysis.")

    # Compute per-scenario stats (using all for dist, but changes only on future)
    def get_top_dist(preds):
        cnt = Counter()
        for p in preds:
            top = p.get("top_5_scorelines", [{}])[0].get("scoreline", "?")
            cnt[top] += 1
        return {k: cnt[k] for k in sorted(cnt, key=cnt.get, reverse=True)}

    def get_one_one_pct(preds):
        if not preds:
            return 0.0
        ones = sum(1 for p in preds if p.get("top_5_scorelines", [{}])[0].get("scoreline") == "1-1")
        return round(ones / len(preds) * 100, 1)

    def get_avg_pred_goals(preds):
        if not preds:
            return 0.0
        totals = [p.get("expected_goals_team_a", 0) + p.get("expected_goals_team_b", 0) for p in preds]
        return round(sum(totals) / len(totals), 3)

    def get_avg_top5_mass(preds):
        if not preds:
            return 0.0
        masses = []
        for p in preds:
            top5 = p.get("top_5_scorelines", [])
            masses.append(sum(s.get("probability", 0) for s in top5))
        return round(sum(masses) / len(masses), 4)

    top_dist_by = {}
    one_one_by = {}
    avg_goals_by = {}
    avg_mass_by = {}
    for p in postures:
        preds = reports[p].get("predictions", [])
        top_dist_by[p] = get_top_dist(preds)
        one_one_by[p] = get_one_one_pct(preds)
        avg_goals_by[p] = get_avg_pred_goals(preds)
        avg_mass_by[p] = get_avg_top5_mass(preds)

    # Changes between postures for future only
    def build_future_lookup(posture):
        lut = {}
        for pr in reports[posture].get("predictions", []):
            if pr["match_id"] not in played_ids:
                top = pr.get("top_5_scorelines", [{}])[0].get("scoreline", "?")
                lut[pr["match_id"]] = {"top": top, "date": pr.get("date"), "team_a": pr.get("team_a"), "team_b": pr.get("team_b")}
        return lut

    future_luts = {p: build_future_lookup(p) for p in postures}

    def diff_list(p1, p2):
        diffs = []
        lut1 = future_luts.get(p1, {})
        lut2 = future_luts.get(p2, {})
        for mid, info in lut1.items():
            info2 = lut2.get(mid, {})
            if info.get("top") != info2.get("top"):
                diffs.append({
                    "match_id": mid,
                    "date": info.get("date"),
                    "team_a": info.get("team_a"),
                    "team_b": info.get("team_b"),
                    f"{p1}_top": info.get("top"),
                    f"{p2}_top": info2.get("top")
                })
        return diffs

    current_to_moderate = diff_list("current", "moderate")
    moderate_to_aggressive = diff_list("moderate", "aggressive")

    # factors
    def get_factor(rep):
        m = rep.get("metadata", {})
        return m.get("calibration_factor") or round( (m.get("calibration_factor_fav",1) + m.get("calibration_factor_und",1))/2 , 3)

    current_factor = get_factor(reports.get("current", {}))
    moderate_factor = get_factor(reports.get("moderate", {}))
    aggressive_factor = get_factor(reports.get("aggressive", {}))

    # risk notes based on audit knowledge
    risk_notes = [
        "Current factor (1.056) is conservative given observed 1.148 raw ratio and persistent high scoring.",
        "Moderate (+0.04) provides a cautious step up without exceeding common caps.",
        "Aggressive (+0.08) approaches raw ratio but carries higher risk of overreacting on n=36 sample; monitor 1X2 stability.",
        "All challengers apply factors only to future matches; past matches use uncalibrated base predictions for honesty.",
        "Do not use aggressive as default until after 8-10 additional results validate it."
    ]

    generated_at = datetime.utcnow().isoformat() + "Z"

    challenger_json = {
        "generated_at": generated_at,
        "current_factor": current_factor,
        "moderate_factor": moderate_factor,
        "aggressive_factor": aggressive_factor,
        "selected_production_posture": "moderate",
        "number_of_future_matches": n_future,
        "top_prediction_distribution_by_scenario": top_dist_by,
        "one_one_percentage_by_scenario": one_one_by,
        "average_predicted_goals_by_scenario": avg_goals_by,
        "average_top5_probability_mass_by_scenario": avg_mass_by,
        "matches_top_changes_current_vs_moderate": current_to_moderate,
        "matches_top_changes_moderate_vs_aggressive": moderate_to_aggressive,
        "risk_notes": risk_notes
    }

    json_path = out_dir / "calibration_challenger_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(challenger_json, f, indent=2)
    print(f"[challenger] Wrote {json_path}")

    # Markdown
    md = f"""# Calibration Challenger Report

Generated: {generated_at}

## Overview
This report compares three live calibration postures applied **only to future matches**:
- current: {current_factor}
- moderate: {moderate_factor}
- aggressive: {aggressive_factor}

Production / default live_predictions.json uses **moderate** posture.

Backtest reports and historical evaluation remain completely uncalibrated (base model).

## Why moderate is the selected production posture
The performance audit showed a raw actual/predicted goal ratio of ~1.148 with 36 evaluated matches (109 actual goals).
The current calibration factor of ~1.056 is a smoothed conservative value (half-step, rolling window 15, bounds).

Moderate (current + ~0.04) steps toward the observed volume gap while remaining within safe bounds (1.10 cap logic) and small-sample caution.

## Why aggressive is not default
Aggressive approaches the raw ratio more closely but risks over-reacting to the current 36-match sample (including possibly anomalous high-scoring games).
We will decide after observing 8-10 additional matches whether to promote aggressive or adjust.

## How to compare scenarios with future results
1. After each matchday, run `python -m src.cli evaluate --actuals-source martj42`
2. Compare the actual goal totals and scorelines against the three live_*.json files (using the future-match predictions that were generated before the matches).
3. Track:
   - Goal volume error per posture
   - Top-1 / Top-5 hit rates per posture (for the matches that were future at generation time)
   - 1-1 concentration shift
   - Whether 1X2 accuracy degrades with higher variance predictions
4. Update production posture in CLI / predict calls when data supports it.

**Important**: Never retro-apply a posture to past matches or backtest reports.

## What to monitor after the next 8-10 matches
- Average actual goals vs the different predicted avgs (current ~2.64, moderate ~2.73?, aggressive ~2.82?)
- Count and % of 1-1 tops in the updated live predictions
- High-scoring matches (5+) frequency and whether they land in top5 more often under higher factors
- Any degradation in aggregate 1X2 accuracy
- Stability of the base calibration computation from calibrate.py

If moderate clearly improves volume without hurting 1X2, consider making it the new "current" base.
Aggressive should only be promoted if data consistently shows underestimation beyond what moderate corrects.
"""

    md_path = out_dir / "calibration_challenger_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[challenger] Wrote {md_path}")


if __name__ == "__main__":
    predict_scorelines()
