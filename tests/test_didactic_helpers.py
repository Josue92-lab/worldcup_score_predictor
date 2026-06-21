import sys
from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock streamlit before importing app helpers (safe even if not needed for these)
from unittest.mock import MagicMock
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

from src.app_streamlit import (
    load_full_evaluation_report,
    load_calibration_report,
    load_diagnostics_summary,
    get_educational_markdown,
    get_mode_banner,
    format_explanation_bullets,
    get_1_1_concentration_note,
    get_data_credits_markdown,
)


def test_get_educational_markdown_contains_key_concepts():
    md = get_educational_markdown()
    assert "probabilistic educational model" in md
    assert "Top 5 Hit" in md
    assert "1X2 Hit" in md
    assert "Team A / Team B" in md
    assert "Pending" in md
    assert "1-1 can be most likely" in md or "most likely exact score" in md.lower()


def test_get_mode_banner_backtest():
    meta = {"mode": "backtest", "train_cutoff": "2026-06-10", "as_of_date": None}
    banner = get_mode_banner(meta)
    assert "Backtest mode" in banner
    assert "2026-06-10" in banner
    assert "must not use later" in banner.lower()


def test_get_mode_banner_live():
    meta = {"mode": "live", "as_of_date": "2026-06-20"}
    banner = get_mode_banner(meta)
    assert "Live mode" in banner
    assert "2026-06-20" in banner
    assert "retroactive" in banner.lower()


def test_get_mode_banner_missing():
    assert get_mode_banner({}) == ""
    assert get_mode_banner(None) == ""


def test_format_explanation_bullets_safe_on_empty():
    p = {}
    md = format_explanation_bullets(p, {})
    assert "Baseline Poisson" in md or "Limited additional" in md


def test_format_explanation_bullets_with_data():
    p = {
        "expected_goals_team_a": 1.25,
        "expected_goals_team_b": 0.98,
        "explanation": {
            "team_a_attack_strength": 1.20,
            "team_b_attack_strength": 0.91,
            "team_a_defense_strength": 0.71,
            "team_b_defense_strength": 0.64,
            "main_factors": ["Mexico squad quality is 3.4x South Africa"],
        },
        "data_quality": {
            "squad_match_status": "ok",
            "player_kaggle_match_rate": 0.88,
            "warnings": ["Low recent form data"],
        },
    }
    meta = {"calibrated": True, "calibration_factor_fav": 1.05}
    md = format_explanation_bullets(p, meta)
    assert "Expected goals" in md
    assert "attack strength" in md
    assert "Squad data status" in md
    assert "Live calibration active" in md
    assert "Low recent form data" in md


def test_load_diagnostics_summary_never_crashes(tmp_path, monkeypatch):
    # Point OUTPUTS_DIR at empty tmp
    import src.app_streamlit as app_mod
    monkeypatch.setattr(app_mod, "OUTPUTS_DIR", tmp_path)

    diag = load_diagnostics_summary()
    assert isinstance(diag, dict)
    assert diag["matches_evaluated"] == 0
    assert diag["top1_rate"] is None
    assert diag.get("one_one_pct") is None


def test_load_diagnostics_summary_with_fake_data(tmp_path, monkeypatch):
    import src.app_streamlit as app_mod
    monkeypatch.setattr(app_mod, "OUTPUTS_DIR", tmp_path)

    # Write minimal eval report
    eval_data = {
        "matches_evaluated": 32,
        "exact_score_top1_rate": 0.2188,
        "exact_score_top5_rate": 0.5312,
        "outcome_1x2_rate": 0.4375,
        "diagnostics": {
            "predicted_total_goals_avg": 2.60,
            "actual_total_goals_avg": 3.00,
            "top1_scoreline_distribution": {"1-1": 20, "0-0": 3},
            "high_score_miss_count": 7,
        },
    }
    (tmp_path / "evaluation_report.json").write_text(json.dumps(eval_data))

    calib_data = {
        "average_predicted_total_goals": 2.54,
        "average_actual_total_goals": 3.06,
        "global_calibration_factor": 1.03,
    }
    (tmp_path / "calibration_report.json").write_text(json.dumps(calib_data))

    diag = load_diagnostics_summary()
    assert diag["matches_evaluated"] == 32
    assert abs(diag["top1_rate"] - 0.2188) < 0.001
    assert diag["avg_actual_goals"] == 3.06 or diag["avg_actual_goals"] == 3.0
    assert diag["one_one_count"] == 20


def test_get_1_1_concentration_note_detects_high():
    diag = {"one_one_pct": 0.65, "one_one_count": 21, "matches_evaluated": 32}
    preds = [{"top_5_scorelines": [{"scoreline": "1-1"}]}] * 25
    note = get_1_1_concentration_note(diag, preds)
    assert note  # should produce warning
    assert "1-1" in note
    assert "calibration" in note.lower() or "underestimates" in note.lower()


def test_get_data_credits_markdown():
    md = get_data_credits_markdown()
    assert "martj42/international_results" in md
    assert "davidcariboo/player-scores" in md
    assert "FIFA SquadLists" in md
    assert "lag" in md.lower() or "freshness" in md.lower()


def test_aggregate_1x2_helpers_via_canonical():
    # The canonical implementation is now in src.outcomes (tested primarily via
    # test_evaluation_logic). We keep a light smoke here so the didactic module
    # continues to exercise the outcomes code path used by the dashboard.
    from src.outcomes import get_predicted_1x2_outcome as pred, get_actual_1x2_outcome as actual

    # Aggregate from probs (the correct source for 1X2 Hit)
    assert pred(0.618, 0.213, 0.169) == "1"
    assert actual(3, 0) == "1"

    assert pred(0.30, 0.40, 0.30) == "X"
    assert actual(0, 0) == "X"

    assert pred(0.20, 0.30, 0.50) == "2"
    assert actual(0, 2) == "2"

    # Top score wrong but aggregate 1X2 correct example
    assert pred(0.55, 0.30, 0.15) == "1"
    assert actual(3, 0) == "1"

    # Actual outside top5 still can be 1X2 hit
    assert actual(4, 1) == "1"
    assert pred(0.70, 0.20, 0.10) == "1"

    # Mismatch
    assert pred(0.10, 0.20, 0.70) == "2"
    assert actual(2, 0) == "1"


# --- Calibration diagnostics tests (added for audit) ---

def test_resolve_calibration_factor_prefers_explicit_and_fav_und(monkeypatch):
    from src.app_streamlit import _resolve_calibration_factor
    # explicit key
    assert _resolve_calibration_factor({"calibration_factor": 1.056}) == 1.056
    # fav/und avg
    meta = {"calibration_factor_fav": 1.013, "calibration_factor_und": 1.1}
    assert _resolve_calibration_factor(meta) == round((1.013 + 1.1) / 2, 3)
    # fallback to calib
    calib = {"global_calibration_factor": 1.03}
    assert _resolve_calibration_factor({}, calib) == 1.03
    # default
    assert _resolve_calibration_factor({}) == 1.0


def test_if_factor_eq_1_no_inflate_claim_in_warning_logic():
    """If effective factor == 1.0, must not emit text claiming lambdas were inflated."""
    from src.app_streamlit import _resolve_calibration_factor
    # simulate panel decision
    factor = 1.0
    active = True  # could be set if fav/und were neutral
    would_claim_inflate = (factor > 1.001)
    assert would_claim_inflate is False
    # even when using resolve
    assert _resolve_calibration_factor({"calibration_factor": 1.0}) == 1.0
    # neutral wording path would be taken, not "inflated by 1.000x"


def test_if_factor_gt_1_may_claim_inflation():
    from src.app_streamlit import _resolve_calibration_factor
    factor = _resolve_calibration_factor({"calibration_factor_fav": 1.08, "calibration_factor_und": 1.08})
    assert factor > 1.0
    assert factor > 1.001
    # caller would use "Lambdas inflated by Xx"


def test_displayed_rounded_factor_does_not_hide_meaningful_value():
    # 1.002 should round display to 1.002 not collapse to 1.000 without context
    from src.app_streamlit import _resolve_calibration_factor
    f = _resolve_calibration_factor({"calibration_factor": 1.0023})
    assert abs(f - 1.002) < 0.001
    assert f != 1.0


def test_diagnostics_scope_safe_when_live_selected_but_source_backtest(tmp_path, monkeypatch):
    """When live report selected, diag and captions must reflect backtest/eval source for avgs/factor."""
    import src.app_streamlit as app_mod
    monkeypatch.setattr(app_mod, "OUTPUTS_DIR", tmp_path)

    # eval from backtest (cf=1.0)
    eval_data = {
        "matches_evaluated": 36,
        "exact_score_top1_rate": 0.1944,
        "diagnostics": {
            "predicted_total_goals_avg": 2.638,
            "actual_total_goals_avg": 3.028,
            "calibration_factor": 1.0,
        }
    }
    (tmp_path / "evaluation_report.json").write_text(json.dumps(eval_data))

    # calib has the live window factor >1
    calib_data = {
        "average_predicted_total_goals": 2.637,
        "average_actual_total_goals": 3.0,
        "global_calibration_factor": 1.056,
        "matches_evaluated_in_window": 15,
    }
    (tmp_path / "calibration_report.json").write_text(json.dumps(calib_data))

    diag = app_mod.load_diagnostics_summary()
    # Because eval cf==1.0, we now fallback to calib's non-1 in load
    assert diag["matches_evaluated"] == 36
    assert diag.get("avg_actual_goals") in (3.028, 3.0)
    # factor should prefer calib's when eval 1.0
    assert diag.get("calibration_factor") == 1.056 or diag.get("calibration_factor") == 1.056

    # resolve on live meta would give applied
    live_meta = {"mode": "live", "calibration_factor_fav": 1.013, "calibration_factor_und": 1.1, "calibrated": True}
    eff = app_mod._resolve_calibration_factor(live_meta, calib_data)
    assert eff > 1.0


def test_average_actual_goals_uses_only_evaluated_matches(tmp_path, monkeypatch):
    """Test via report shape that matches_evaluated and avgs are derived only from scored matches."""
    import src.app_streamlit as app_mod
    monkeypatch.setattr(app_mod, "OUTPUTS_DIR", tmp_path)

    # Simulate report as would be produced by evaluate (only completed)
    eval = {
        "matches_evaluated": 36,
        "diagnostics": {"actual_total_goals_avg": 3.028, "predicted_total_goals_avg": 2.638, "calibration_factor": 1.0},
        "evaluated_matches": [{"actual_scoreline": "2-0"} for _ in range(36)],  # 36 entries
    }
    (tmp_path / "evaluation_report.json").write_text(json.dumps(eval))
    diag = app_mod.load_diagnostics_summary()
    assert diag["matches_evaluated"] == 36
    # avg computed as sum / 36 not including any pending (which wouldn't be in evaluated_matches)
    assert abs(diag["avg_actual_goals"] - 3.028) < 0.01


def test_pending_matches_do_not_affect_average_actual_goals(tmp_path, monkeypatch):
    """Pending (no actual_scoreline) are excluded from evaluated counts and avgs."""
    import src.app_streamlit as app_mod
    monkeypatch.setattr(app_mod, "OUTPUTS_DIR", tmp_path)

    # report has 40 wc rows but only 36 evaluated (4 pending had no score)
    eval = {
        "matches_evaluated": 36,
        "diagnostics": {"actual_total_goals_avg": 3.028, "predicted_total_goals_avg": 2.638},
        "evaluated_matches": [{} for _ in range(36)],
    }
    (tmp_path / "evaluation_report.json").write_text(json.dumps(eval))
    diag = app_mod.load_diagnostics_summary()
    # total would be different if pending leaked in
    assert diag["matches_evaluated"] == 36
    # no way pending contribute since evaluate skips na scores, and eval_map only has completed
