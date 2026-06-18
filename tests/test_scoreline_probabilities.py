import pytest
import numpy as np
from src.model_poisson import calculate_match_probabilities

def test_probabilities_non_negative():
    result = calculate_match_probabilities(1.5, 1.2, max_goals=5)
    assert np.all(result["matrix"] >= 0)

def test_probabilities_sum_to_one():
    result = calculate_match_probabilities(1.5, 1.2, max_goals=10)
    assert pytest.approx(result["matrix"].sum(), 0.01) == 1.0

def test_top_5_descending():
    result = calculate_match_probabilities(1.5, 1.2, max_goals=5)
    probs = [x['probability'] for x in result['top_5']]
    assert probs == sorted(probs, reverse=True)
    assert len(probs) == 5

def test_deterministic():
    res1 = calculate_match_probabilities(2.0, 1.0, max_goals=5)
    res2 = calculate_match_probabilities(2.0, 1.0, max_goals=5)
    
    assert res1['top_5'] == res2['top_5']
