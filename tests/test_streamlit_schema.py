import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.app_streamlit import normalize_prediction

def test_normalize_list_schema_backward_compatibility():
    # Format 1: Just a list of bare minimal fields
    raw_row = {"team_a": "Argentina", "team_b": "Austria"}
    norm = normalize_prediction(raw_row)
    
    assert norm["team_a"] == "Argentina"
    assert norm["team_b"] == "Austria"
    assert norm["home_team"] == "Argentina"
    assert norm["away_team"] == "Austria"
    assert norm["team_a_label"] == "Team A / Listed first"
    assert norm["neutral"] is True
    assert norm["venue"] == "Unknown"

def test_normalize_dict_schema():
    # Format 2: Full schema with all fields
    raw_row = {
        "team_a": "Argentina",
        "team_b": "Austria",
        "home_team": "Argentina",
        "away_team": "Austria",
        "team_a_label": "Team A / Listed first",
        "team_b_label": "Team B / Listed second",
        "neutral": True,
        "venue": "Lusail",
        "top_5_scorelines": [
            {
                "team_a_goals": 2,
                "team_b_goals": 1,
                "scoreline": "2-1",
                "display_scoreline": "Argentina 2 - 1 Austria",
                "probability": 0.097
            }
        ]
    }
    norm = normalize_prediction(raw_row)
    
    assert norm["team_a"] == "Argentina"
    assert norm["team_b"] == "Austria"
    assert norm["venue"] == "Lusail"
    assert len(norm["top_5_scorelines"]) == 1
    assert norm["top_5_scorelines"][0]["display_scoreline"] == "Argentina 2 - 1 Austria"
