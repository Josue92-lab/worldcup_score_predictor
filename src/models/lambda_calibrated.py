"""
Lambda calibrated Poisson - global goal volume multiplier.
"""

from .base_poisson import BasePoissonModel
from src.model_poisson import calculate_match_probabilities
from src.normalize_teams import normalize_team_name
from src.outcomes import get_predicted_1x2_outcome

class LambdaCalibratedModel(BasePoissonModel):
    """Poisson with global lambda multiplier. Default ~1.15 for moderate volume correction."""

    def __init__(self, factor=1.15, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.model_version = f"lambda_{factor:.2f}"

    def predict(self, match: dict, context: dict = None) -> dict:
        base_pred = super().predict(match, context) if self.model else None

        if self.model is None:
            # fallback simple if not fitted - use from context or default
            # For production, caller should fit
            xga = match.get("base_xga", 1.3)
            xgb = match.get("base_xgb", 1.3)
        else:
            team_a = normalize_team_name(match["team_a"])
            team_b = normalize_team_name(match["team_b"])
            xga, xgb = self._predict_xg(team_a, team_b)

        nxga = xga * self.factor
        nxgb = xgb * self.factor

        probs = calculate_match_probabilities(nxga, nxgb, self.max_goals, self.rho)

        top5 = probs["top_5"]
        top_score = top5[0]["scoreline"] if top5 else "1-1"
        top_p = top5[0]["probability"] if top5 else 0.0

        wa, wd, wb = probs["win_a"], probs["draw"], probs["win_b"]
        p1x2 = get_predicted_1x2_outcome(wa, wd, wb)

        top5_list = [{"scoreline": s["scoreline"], "probability": round(s["probability"], 4)} for s in top5]

        total = round(nxga + nxgb, 3)

        legacy = base_pred if base_pred else {}

        return {
            "model_version": self.model_version,
            "scoreline_engine": "lambda_calibrated_poisson",
            "outcome_engine": "lambda_calibrated_poisson",
            "goal_volume_policy": f"lambda_{self.factor}",
            "top5_scorelines": top5_list,
            "top_prediction": top_score,
            "team_a_win_probability": round(wa, 4),
            "draw_probability": round(wd, 4),
            "team_b_win_probability": round(wb, 4),
            "predicted_aggregate_1x2": p1x2,
            "expected_team_a_goals": round(nxga, 3),
            "expected_team_b_goals": round(nxgb, 3),
            "predicted_total_goals": total,
            "legacy_base_comparison": {
                "base_top_prediction": legacy.get("top_prediction"),
                "base_predicted_total": legacy.get("predicted_total_goals"),
            },
            "model_notes": [f"Global lambda {self.factor} applied for volume correction."]
        }

    def _predict_xg(self, team_a, team_b):
        from src.model_poisson import predict_xg
        return predict_xg(team_a, team_b, self.model)
