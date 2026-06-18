import pytest
import json
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.evaluate_predictions as ev

def test_evaluation_1x2_logic():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup mock directories
        outputs_dir = temp_path / "outputs"
        raw_dir = temp_path / "raw"
        outputs_dir.mkdir()
        (raw_dir / "international_results").mkdir(parents=True)
        
        # 1. Create mock predictions
        mock_predictions = {
            "predictions": [
                {
                    "match_id": "M1",
                    "date": "2026-06-11",
                    "team_a": "Team A",
                    "team_b": "Team B",
                    "top_5_scorelines": [{"scoreline": "1-1"}]
                },
                {
                    "match_id": "M2",
                    "date": "2026-06-12",
                    "team_a": "Team C",
                    "team_b": "Team D",
                    "top_5_scorelines": [{"scoreline": "2-0"}]
                },
                {
                    "match_id": "M3",
                    "date": "2026-06-13",
                    "team_a": "Team E",
                    "team_b": "Team F",
                    "top_5_scorelines": [{"scoreline": "1-0"}]
                }
            ]
        }
        
        with open(outputs_dir / "backtest_report.json", "w") as f:
            json.dump(mock_predictions, f)
            
        # 2. Create mock actuals
        mock_actuals = pd.DataFrame([
            {
                "date": "2026-06-11",
                "home_team": "Team A",
                "away_team": "Team B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "World Cup"
            },
            {
                "date": "2026-06-12",
                "home_team": "Team C",
                "away_team": "Team D",
                "home_score": 1,
                "away_score": 0,
                "tournament": "World Cup"
            },
            {
                "date": "2026-06-13",
                "home_team": "Team E",
                "away_team": "Team F",
                "home_score": 0,
                "away_score": 1,
                "tournament": "World Cup"
            }
        ])
        mock_actuals.to_csv(raw_dir / "international_results" / "results.csv", index=False)
        
        # 3. Patch and run
        with patch.object(ev, 'OUTPUTS_DIR', outputs_dir), \
             patch.object(ev, 'RAW_DIR', raw_dir):
            ev.evaluate_predictions()
            
            # 4. Verify results
            eval_report_path = outputs_dir / "evaluation_report.json"
            assert eval_report_path.exists()
            
            with open(eval_report_path, "r") as f:
                report = json.load(f)
                
            matches = report["evaluated_matches"]
            assert len(matches) == 3
            
            # M1: Pred 1-1, Act 1-1 => Draw, Draw => True
            m1 = next(m for m in matches if m["match_id"] == "M1")
            assert m1["outcome_1x2_correct"] is True
            
            # M2: Pred 2-0, Act 1-0 => Home Win, Home Win => True
            m2 = next(m for m in matches if m["match_id"] == "M2")
            assert m2["outcome_1x2_correct"] is True
            
            # M3: Pred 1-0, Act 0-1 => Home Win, Away Win => False
            m3 = next(m for m in matches if m["match_id"] == "M3")
            assert m3["outcome_1x2_correct"] is False
