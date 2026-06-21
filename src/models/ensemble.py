"""
Ensemble model - best scoreline coverage from bake-off.
Averages multiple engines transparently.
"""

from .base_poisson import BasePoissonModel
from .hybrid_elo_poisson import HybridEloPoissonModel
from .lambda_calibrated import LambdaCalibratedModel
from src.model_poisson import calculate_match_probabilities
from src.normalize_teams import normalize_team_name
from src.outcomes import get_predicted_1x2_outcome

class EnsembleModel:
    """Ensemble of strong components for scoreline quality."""

    def __init__(self, components=None, **kwargs):
        self.components = components or ["base", "hybrid", "lambda"]
        self.base = BasePoissonModel(**kwargs)
        self.hybrid = HybridEloPoissonModel(**kwargs)
        self.lambda_m = LambdaCalibratedModel(factor=1.15, **kwargs)
        self.model_version = "ensemble"

    def fit(self, historical_df, results_df=None):
        self.base.fit(historical_df)
        if results_df is not None:
            self.hybrid.fit(historical_df, results_df)
        self.lambda_m.fit(historical_df)
        return self

    def predict(self, match: dict, context: dict = None) -> dict:
        if self.base.model is None:
            raise RuntimeError("Fit first.")

        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        # Collect xG from components
        p_xga, p_xgb = self.base._predict_xg(team_a, team_b) if hasattr(self.base, '_predict_xg') else (1.3, 1.3)  # rough
        # better use the predict_xg
        from src.model_poisson import predict_xg
        bxga, bxgb = predict_xg(team_a, team_b, self.base.model)
        lxga, lxgb = bxga * 1.15, bxgb * 1.15   # lambda 1.15

        h_rec = self.hybrid.predict(match)
        hxga = h_rec["expected_team_a_goals"]
        hxgb = h_rec["expected_team_b_goals"]

        # Average xG for scoreline (good coverage)
        xga = (bxga + lxga + hxga) / 3.0
        xgb = (bxgb + lxgb + hxgb) / 3.0

        probs = calculate_match_probabilities(xga, xgb, self.base.max_goals, self.base.rho)

        top5 = probs["top_5"]
        top_sl = top5[0]["scoreline"] if top5 else "1-1"
        top_p = top5[0]["probability"] if top5 else 0.0

        wa, wd, wb = probs["win_a"], probs["draw"], probs["win_b"]
        p1x2 = get_predicted_1x2_outcome(wa, wd, wb)  # but we'll override with hybrid for core_v2

        top5_list = [{"scoreline": s["scoreline"], "probability": round(s["probability"], 4)} for s in top5]

        total = round(xga + xgb, 3)

        return {
            "model_version": self.model_version,
            "scoreline_engine": "ensemble",
            "outcome_engine": "ensemble",  # will be overridden in core_v2
            "goal_volume_policy": "ensemble_average",
            "top5_scorelines": top5_list,
            "top_prediction": top_sl,
            "team_a_win_probability": round(wa, 4),
            "draw_probability": round(wd, 4),
            "team_b_win_probability": round(wb, 4),
            "predicted_aggregate_1x2": p1x2,
            "expected_team_a_goals": round(xga, 3),
            "expected_team_b_goals": round(xgb, 3),
            "predicted_total_goals": total,
            "legacy_base_comparison": {"base_top": top_sl},
            "model_notes": ["Ensemble average xG for best scoreline coverage (Top-5). 1X2 will be replaced by hybrid in core_v2."]
        }
