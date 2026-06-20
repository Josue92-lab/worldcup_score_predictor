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
                    # Brazil vs Haiti
                    "match_id": "M1",
                    "date": "2026-06-11",
                    "team_a": "Brazil",
                    "team_b": "Haiti",
                    "team_a_win_probability": 0.618,
                    "draw_probability": 0.213,
                    "team_b_win_probability": 0.169,
                    "top_5_scorelines": [{"scoreline": "2-0"}, {"scoreline": "1-0"}, {"scoreline": "2-1"}, {"scoreline": "1-1"}, {"scoreline": "3-1"}]
                },
                {
                    # Draw vs Draw
                    "match_id": "M2",
                    "date": "2026-06-12",
                    "team_a": "Team C",
                    "team_b": "Team D",
                    "team_a_win_probability": 0.300,
                    "draw_probability": 0.400,
                    "team_b_win_probability": 0.300,
                    "top_5_scorelines": [{"scoreline": "1-1"}]
                },
                {
                    # Team B vs Team A Actual
                    "match_id": "M3",
                    "date": "2026-06-13",
                    "team_a": "Team E",
                    "team_b": "Team F",
                    "team_a_win_probability": 0.200,
                    "draw_probability": 0.300,
                    "team_b_win_probability": 0.500,
                    "top_5_scorelines": [{"scoreline": "0-1"}]
                }
            ]
        }
        
        with open(outputs_dir / "backtest_report.json", "w") as f:
            json.dump(mock_predictions, f)
            
        # 2. Create mock actuals
        mock_actuals = pd.DataFrame([
            {
                "date": "2026-06-11",
                "home_team": "Brazil",
                "away_team": "Haiti",
                "home_score": 3,
                "away_score": 0,
                "tournament": "World Cup"
            },
            {
                "date": "2026-06-12",
                "home_team": "Team C",
                "away_team": "Team D",
                "home_score": 0,
                "away_score": 0,
                "tournament": "World Cup"
            },
            {
                "date": "2026-06-13",
                "home_team": "Team E",
                "away_team": "Team F",
                "home_score": 2,
                "away_score": 0,
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
            
            # M1: Brazil vs Haiti (Pred 1 Win, Act 3-0 Win) => True
            m1 = next(m for m in matches if m["match_id"] == "M1")
            assert m1["outcome_1x2_correct"] is True
            assert m1["top1_correct"] is False
            assert m1["top5_correct"] is False
            
            # M2: Draw vs Draw => True
            m2 = next(m for m in matches if m["match_id"] == "M2")
            assert m2["outcome_1x2_correct"] is True
            
            # M3: Pred B Win, Act A Win => False
            m3 = next(m for m in matches if m["match_id"] == "M3")
            assert m3["outcome_1x2_correct"] is False
