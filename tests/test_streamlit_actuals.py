import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from unittest import mock
import pytest
from src.app_streamlit import main

@mock.patch("src.app_streamlit.st")
@mock.patch("src.app_streamlit.OUTPUTS_DIR")
@mock.patch("src.app_streamlit.FIXTURE_PATH")
def test_streamlit_evaluation_merge(mock_fixture_path, mock_outputs_dir, mock_st, tmp_path):
    mock_outputs_dir.return_value = tmp_path
    
    # Setup predictions.json
    preds = {
        "metadata": {"mode": "predict"},
        "predictions": [
            {
                "match_id": "M_1",
                "date": "2026-06-15",
                "team_a": "Mexico",
                "team_b": "South Africa",
                "top_5_scorelines": [
                    {"scoreline": "1-1", "probability": 0.1515},
                    {"scoreline": "0-0", "probability": 0.1396},
                    {"scoreline": "1-0", "probability": 0.1211},
                    {"scoreline": "0-1", "probability": 0.1009},
                    {"scoreline": "2-0", "probability": 0.0785}
                ]
            },
            {
                "match_id": "M_2",
                "date": "2026-06-16",
                "team_a": "England",
                "team_b": "Croatia",
                "top_5_scorelines": [
                    {"scoreline": "1-0", "probability": 0.15},
                    {"scoreline": "2-0", "probability": 0.10}
                ]
            }
        ]
    }
    
    preds_path = tmp_path / "predictions.json"
    with open(preds_path, "w") as f:
        json.dump(preds, f)
        
    # Setup evaluation_report.json
    evals = {
        "evaluated_matches": [
            {
                "match_id": "M_1",
                "actual_scoreline": "2-0",
                "top1_correct": False,
                "top5_correct": True,
                "outcome_1x2_correct": True
            },
            {
                "match_id": "M_2",
                "actual_scoreline": "4-2",
                "top1_correct": False,
                "top5_correct": False,
                "outcome_1x2_correct": True
            }
        ]
    }
    
    eval_path = tmp_path / "evaluation_report.json"
    with open(eval_path, "w") as f:
        json.dump(evals, f)

    # Instead of patching OUTPUTS_DIR directly (since it's imported from src.config),
    # let's mock the file existence and reading.
    pass
