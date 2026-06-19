import sys
import json
from pathlib import Path
import pytest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_streamlit import load_evaluation_report, normalize_prediction

def test_streamlit_actuals_merge_smoke(tmp_path, monkeypatch):
    # Mock OUTPUTS_DIR in app_streamlit to point to tmp_path
    monkeypatch.setattr("src.app_streamlit.OUTPUTS_DIR", tmp_path)
    
    # 1. Setup evaluation_report.json
    evals = {
        "evaluated_matches": [
            {
                "match_id": "GS_001",
                "actual_scoreline": "2-0",
                "top5_correct": True,
                "outcome_1x2_correct": True
            }
        ]
    }
    
    eval_path = tmp_path / "evaluation_report.json"
    with open(eval_path, "w") as f:
        json.dump(evals, f)

    # 2. Selected report row
    raw_row = {
        "match_id": "GS_001",
        "team_a": "Mexico",
        "team_b": "South Africa",
        "date": "2026-06-11",
        "top_5_scorelines": [
            {"scoreline": "1-1", "display_scoreline": "Mexico 1 - 1 South Africa"},
            {"scoreline": "2-0", "display_scoreline": "Mexico 2 - 0 South Africa"}
        ]
    }
    
    p = normalize_prediction(raw_row)
    
    # 3. Replicate the merge logic from app_streamlit.py
    eval_map = load_evaluation_report()
    
    match_id = str(p.get("match_id"))
    eval_match = eval_map.get(match_id)
    
    if not eval_match:
        from src.normalize_teams import normalize_team_name
        p_date_str = str(p.get("date")) if p.get("date") else ""
        ta = normalize_team_name(p.get("team_a", ""))
        tb = normalize_team_name(p.get("team_b", ""))
        eval_match = eval_map.get(f"{p_date_str}_{ta}_{tb}")
        
    p["_actual_score"] = eval_match.get("actual_scoreline") if eval_match else None
    p["_top_5_hit"] = eval_match.get("top5_correct") if eval_match else None
    p["_1x2_hit"] = eval_match.get("outcome_1x2_correct") if eval_match else None
    
    # 4. Assertions
    assert p["_actual_score"] == "2-0"
    assert p["_top_5_hit"] is True
    assert p["_1x2_hit"] is True
    
    # Replicate Top 5 table Result marking
    import pandas as pd
    scoreline_data = pd.DataFrame(p['top_5_scorelines'])
    actual_score = p.get("_actual_score")
    
    scoreline_data["Result"] = scoreline_data["scoreline"].apply(
        lambda x: "Actual result" if x == actual_score else ""
    )
    
    results = scoreline_data["Result"].tolist()
    assert results == ["", "Actual result"]
