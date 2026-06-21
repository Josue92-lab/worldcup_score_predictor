"""
Hybrid Elo + Poisson model.
Best 1X2 from bake-off.
"""

from .base_poisson import BasePoissonModel
from .elo import EloModel
from src.model_poisson import predict_xg, calculate_match_probabilities
from src.normalize_teams import normalize_team_name
from src.outcomes import get_predicted_1x2_outcome

class HybridEloPoissonModel:
    """Blend of base Poisson xG and Elo strength differential. Best 1X2 engine."""

    def __init__(self, blend=0.5, **kwargs):
        self.blend = blend  # weight on elo vs poisson
        self.base_model = BasePoissonModel(**kwargs)
        self.elo_model = EloModel(**kwargs)
        self.model_version = "hybrid_elo_poisson"

    def fit(self, historical_df, results_df):
        self.base_model.fit(historical_df)
        self.elo_model.fit(results_df)
        return self

    def predict(self, match: dict, context: dict = None) -> dict:
        if self.base_model.model is None:
            raise RuntimeError("Fit models first with pre-cutoff data.")

        team_a = normalize_team_name(match.get("team_a"))
        team_b = normalize_team_name(match.get("team_b"))

        p_xga, p_xgb = predict_xg(team_a, team_b, self.base_model.model)
        e_rec = self.elo_model.predict(match)
        e_xga = e_rec["expected_team_a_goals"]
        e_xgb = e_rec["expected_team_b_goals"]

        # Average xG
        xga = (p_xga * (1 - self.blend) + e_xga * self.blend)
        xgb = (p_xgb * (1 - self.blend) + e_xgb * self.blend)

        probs = calculate_match_probabilities(xga, xgb, self.base_model.max_goals, self.base_model.rho)

        top5 = probs["top_5"]
        top_sl = top5[0]["scoreline"] if top5 else "1-1"
        top_p = top5[0]["probability"] if top5 else 0.0
        wa, wd, wb = probs["win_a"], probs["draw"], probs["win_b"]
        p1x2 = get_predicted_1x2_outcome(wa, wd, wb)

        top5_list = [{"scoreline": s["scoreline"], "probability": round(s["probability"], 4)} for s in top5]

        total = round(xga + xgb, 3)

        base_comp = {
            "base_top": top_sl,  # for comparison
            "base_total": total
        }

        return {
            "model_version": self.model_version,
            "scoreline_engine": "hybrid_elo_poisson",
            "outcome_engine": self.model_version,
            "goal_volume_policy": "none",
            "top5_scorelines": top5_list,
            "top_prediction": top_sl,
            "team_a_win_probability": round(wa, 4),
            "draw_probability": round(wd, 4),
            "team_b_win_probability": round(wb, 4),
            "predicted_aggregate_1x2": p1x2,
            "expected_team_a_goals": round(xga, 3),
            "expected_team_b_goals": round(xgb, 3),
            "predicted_total_goals": total,
            "legacy_base_comparison": base_comp,
            "model_notes": [f"Hybrid Elo+Poisson (blend={self.blend}) - best 1X2 from bake-off."]
        }
