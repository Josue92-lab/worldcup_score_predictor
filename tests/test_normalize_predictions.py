import sys
import pytest
from unittest.mock import MagicMock

# Mock streamlit before importing the app so it doesn't execute st.set_page_config
mock_st = MagicMock()
mock_st.tabs.return_value = (MagicMock(), MagicMock(), MagicMock())
mock_st.columns.side_effect = lambda n: tuple(MagicMock() for _ in range(n))
sys.modules['streamlit'] = mock_st

# Now we can safely import the normalizer from the app
from src.app_streamlit import get_first, normalize_prediction

def test_get_first():
    row = {"team_a": "Brazil", "team1": "Argentina"}
    assert get_first(row, "home_team", "team_a", "team1") == "Brazil"
    assert get_first(row, "home_team", "team1", "team_a") == "Argentina"
    assert get_first(row, "missing", default="Fallback") == "Fallback"

def test_normalize_prediction_with_team_a_b():
    # Simulate a JSON row that uses team_a / team_b instead of home/away
    raw_row = {
        "team_a": "France",
        "team_b": "Germany",
        "top_scores": [{"scoreline": "1-0", "probability": 0.1}],
        "team_a_win_probability": 0.45,
        "team_b_win_probability": 0.30,
        "draw_probability": 0.25
    }
    
    normalized = normalize_prediction(raw_row)
    
    assert normalized["home_team"] == "France"
    assert normalized["away_team"] == "Germany"
    assert normalized["top_5_scorelines"] == [{"scoreline": "1-0", "probability": 0.1}]
    assert normalized["home_win_probability"] == 0.45
    assert normalized["away_win_probability"] == 0.30
    # Original keys should still exist
    assert normalized["team_a"] == "France"

def test_normalize_prediction_with_home_away():
    # Simulate a JSON row that already uses the desired schema
    raw_row = {
        "home_team": "Spain",
        "away_team": "Italy",
        "top_5_scorelines": [{"scoreline": "2-1", "probability": 0.15}],
        "home_win_probability": 0.50,
        "away_win_probability": 0.20
    }
    
    normalized = normalize_prediction(raw_row)
    
    assert normalized["home_team"] == "Spain"
    assert normalized["away_team"] == "Italy"
    assert normalized["top_5_scorelines"] == [{"scoreline": "2-1", "probability": 0.15}]
    assert normalized["home_win_probability"] == 0.50
    assert normalized["away_win_probability"] == 0.20
