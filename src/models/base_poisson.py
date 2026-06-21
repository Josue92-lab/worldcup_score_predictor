"""
Base Poisson model - legacy implementation.

Wraps the original fit_poisson_model + predict_xg + calculate_match_probabilities.
Used for honest backtests and as benchmark.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_poisson import fit_poisson_model, predict_xg, calculate_match_probabilities
from src.normalize_teams import normalize_team_name
from src.outcomes import get_predicted_1x2_outcome

class BasePoissonModel:
    """Legacy baseline model. No post-cutoff calibration."""

    def __init__(self, historical_df=None, max_goals=7, rho=-0.13, **kwargs):
        self.max_goals = max_goals
        self.rho = rho
        self.model = None
        self.historical_df = historical_df

    def fit(self, historical_df):
        """Fit using only pre-cutoff data. Caller must filter."""
        self.historical_df = historical_df
        self.model = fit_poisson_model(historical_df)
        return self

    def predict(self, match: dict, context: dict = None) -> dict:
        """
        Predict for a single match using base model.
        Returns structure compatible with Core v2.
        """
        if self.model is None:
            # Try to load from backtest if available, but for production use caller must fit
            raise RuntimeError("Model not fitted. Use fit() with pre-cutoff data or provide in context.")

        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        xga, xgb = predict_xg(team_a, team_b, self.model)

        probs = calculate_match_probabilities(xga, xgb, self.max_goals, self.rho)

        top5 = probs["top_5"]
        top_score = top5[0]["scoreline"] if top5 else "1-1"
        top_p = top5[0]["probability"] if top5 else 0.0

        wa = probs["win_a"]
        wd = probs["draw"]
        wb = probs["win_b"]

        p1x2 = get_predicted_1x2_outcome(wa, wd, wb)

        top5_list = [
            {"scoreline": s["scoreline"], "probability": round(s["probability"], 4)}
            for s in top5
        ]

        total = round(xga + xgb, 3)

        return {
            "model_version": "base",
            "scoreline_engine": "base_poisson",
            "outcome_engine": "base_poisson",
            "goal_volume_policy": "none",
            "top5_scorelines": top5_list,
            "top_prediction": top_score,
            "team_a_win_probability": round(wa, 4),
            "draw_probability": round(wd, 4),
            "team_b_win_probability": round(wb, 4),
            "predicted_aggregate_1x2": p1x2,
            "expected_team_a_goals": round(xga, 3),
            "expected_team_b_goals": round(xgb, 3),
            "predicted_total_goals": total,
            "legacy_base_comparison": None,
            "model_notes": ["Legacy baseline (pre-cutoff only). Used for benchmark."]
        }
