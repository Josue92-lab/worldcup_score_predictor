"""
Core v2 Hybrid Predictor.

Scoreline engine: ensemble (best Top-5)
1X2 engine: hybrid_elo_poisson (best Brier/1X2)
Goal volume: moderate lambda ~1.15
Includes legacy base comparison and divergence diagnostics.
"""

from .ensemble import EnsembleModel
from .hybrid_elo_poisson import HybridEloPoissonModel
from .base_poisson import BasePoissonModel
from src.model_poisson import predict_xg, calculate_match_probabilities
from src.normalize_teams import normalize_team_name
from src.outcomes import get_predicted_1x2_outcome

class CoreV2Predictor:
    """
    Production Core v2.
    Reconciles different engines transparently.
    """

    def __init__(self, lambda_factor=1.15, **kwargs):
        self.lambda_factor = lambda_factor
        self.ensemble = EnsembleModel(**kwargs)
        self.hybrid = HybridEloPoissonModel(blend=0.5, **kwargs)
        self.base = BasePoissonModel(**kwargs)
        self.model_version = "core_v2"

    def fit(self, historical_df, results_df=None):
        self.base.fit(historical_df)
        self.ensemble.fit(historical_df, results_df)
        if results_df is not None:
            self.hybrid.fit(historical_df, results_df)
        return self

    def predict(self, match: dict, context: dict = None) -> dict:
        """
        Unified Core v2 prediction.
        """
        if self.base.model is None:
            raise RuntimeError("Core v2 must be fitted with pre-cutoff data.")

        team_a = normalize_team_name(match["team_a"])
        team_b = normalize_team_name(match["team_b"])

        # 1. Ensemble for scorelines (best coverage)
        ens = self.ensemble.predict(match, context)
        top5 = ens["top5_scorelines"]
        top_prediction = ens["top_prediction"]
        top_p = ens.get("top_prediction_probability", top5[0]["probability"] if top5 else 0.15)
        top5_mass = sum(s["probability"] for s in top5)

        # Apply moderate lambda to expected goals for volume
        base_xga, base_xgb = predict_xg(team_a, team_b, self.base.model)
        xga = base_xga * self.lambda_factor
        xgb = base_xgb * self.lambda_factor

        # Recompute scoreline probs with calibrated xG (ensemble spirit)
        # For simplicity we use the averaged + calibrated
        probs_cal = calculate_match_probabilities(xga, xgb, self.base.max_goals, self.base.rho)
        cal_top5 = [{"scoreline": s["scoreline"], "probability": round(s["probability"], 4)} 
                    for s in probs_cal["top_5"]]
        # Prefer ensemble list if better coverage, else calibrated; here we use cal for volume + ens structure
        # But task says use ensemble for scoreline. We keep ensemble top5 but adjust totals.
        # For transparency, use ensemble top5 but note volume from lambda.
        scoreline_top5 = top5  # from ensemble
        scoreline_top = top_prediction

        # 2. Hybrid for 1X2 (best 1X2)
        hyb = self.hybrid.predict(match, context)
        hyb_wa = hyb["team_a_win_probability"]
        hyb_wd = hyb["draw_probability"]
        hyb_wb = hyb["team_b_win_probability"]
        hyb_1x2 = hyb["predicted_aggregate_1x2"]

        # Reconcile: use hybrid 1X2, ensemble/calibrated scorelines
        # Compute scoreline-derived 1x2 for divergence
        score_wa = probs_cal["win_a"]
        score_wd = probs_cal["draw"]
        score_wb = probs_cal["win_b"]
        score_1x2 = get_predicted_1x2_outcome(score_wa, score_wd, score_wb)

        div = round(abs(hyb_wa - score_wa) + abs(hyb_wd - score_wd) + abs(hyb_wb - score_wb), 4)
        divergence_warning = div > 0.15

        total = round(xga + xgb, 3)

        # Legacy base for comparison
        base_pred = self.base.predict(match, context)

        return {
            "model_version": self.model_version,
            "scoreline_engine": "ensemble",
            "outcome_engine": "hybrid_elo_poisson",
            "goal_volume_policy": f"lambda_{self.lambda_factor}",
            "top5_scorelines": scoreline_top5,
            "top_prediction": scoreline_top,
            "team_a_win_probability": round(hyb_wa, 4),
            "draw_probability": round(hyb_wd, 4),
            "team_b_win_probability": round(hyb_wb, 4),
            "predicted_aggregate_1x2": hyb_1x2,
            "expected_team_a_goals": round(xga, 3),
            "expected_team_b_goals": round(xgb, 3),
            "predicted_total_goals": total,
            "legacy_base_comparison": {
                "base_top_prediction": base_pred.get("top_prediction"),
                "base_predicted_total_goals": base_pred.get("predicted_total_goals"),
                "base_1x2": base_pred.get("predicted_aggregate_1x2"),
            },
            "scoreline_derived_1x2": {
                "wa": round(score_wa, 4),
                "wd": round(score_wd, 4),
                "wb": round(score_wb, 4),
                "p1x2": score_1x2
            },
            "hybrid_1x2": {
                "wa": round(hyb_wa, 4),
                "wd": round(hyb_wd, 4),
                "wb": round(hyb_wb, 4),
                "p1x2": hyb_1x2
            },
            "1x2_divergence": div,
            "divergence_warning": divergence_warning,
            "model_notes": [
                f"Core v2: ensemble for scorelines (Top-5), hybrid_elo_poisson for 1X2, lambda {self.lambda_factor} volume.",
                "Divergence between engines is surfaced for transparency."
            ]
        }
