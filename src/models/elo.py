"""
Simple Elo based model.
Pre-cutoff only.
"""

import pandas as pd
from collections import defaultdict
from src.normalize_teams import normalize_team_name
from src.model_poisson import calculate_match_probabilities
from src.outcomes import get_predicted_1x2_outcome

class EloModel:
    """Basic Elo rating converted to expected goals and probabilities."""

    def __init__(self, k=20, base_rating=1500, max_goals=7, rho=-0.13, base_goals=2.6):
        self.k = k
        self.base_rating = base_rating
        self.max_goals = max_goals
        self.rho = rho
        self.base_goals = base_goals
        self.ratings = defaultdict(lambda: base_rating)

    def fit(self, results_df: pd.DataFrame):
        """Fit using only results <= cutoff."""
        res = results_df.dropna(subset=["home_score", "away_score"]).copy()
        res = res.sort_values("date")
        for _, r in res.iterrows():
            ha = normalize_team_name(r["home_team"])
            hb = normalize_team_name(r["away_team"])
            sa, sb = int(r["home_score"]), int(r["away_score"])
            ra, rb = self.ratings[ha], self.ratings[hb]
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 - ea
            margin = min(abs(sa - sb), 3) / 3.0 + 1
            sa_out = 1 if sa > sb else (0.5 if sa == sb else 0)
            sb_out = 1 - sa_out
            self.ratings[ha] = ra + self.k * margin * (sa_out - ea)
            self.ratings[hb] = rb + self.k * margin * (sb_out - eb)
        return self

    def predict(self, match: dict, context: dict = None) -> dict:
        ta = normalize_team_name(match["team_a"])
        tb = normalize_team_name(match["team_b"])
        ra = self.ratings.get(ta, self.base_rating)
        rb = self.ratings.get(tb, self.base_rating)
        diff = ra - rb
        exp_diff = diff / 200.0
        xga = max(0.5, self.base_goals / 2 + exp_diff * 0.6)
        xgb = max(0.5, self.base_goals / 2 - exp_diff * 0.6)

        probs = calculate_match_probabilities(xga, xgb, self.max_goals, self.rho)
        top5 = probs["top_5"]
        top_sl = top5[0]["scoreline"] if top5 else "1-1"
        top_p = top5[0]["probability"] if top5 else 0.0

        wa, wd, wb = probs["win_a"], probs["draw"], probs["win_b"]
        p1x2 = get_predicted_1x2_outcome(wa, wd, wb)

        top5_list = [{"scoreline": s["scoreline"], "probability": round(s["probability"], 4)} for s in top5]

        return {
            "model_version": "elo",
            "scoreline_engine": "elo",
            "outcome_engine": "elo",
            "goal_volume_policy": "none",
            "top5_scorelines": top5_list,
            "top_prediction": top_sl,
            "team_a_win_probability": round(wa, 4),
            "draw_probability": round(wd, 4),
            "team_b_win_probability": round(wb, 4),
            "predicted_aggregate_1x2": p1x2,
            "expected_team_a_goals": round(xga, 3),
            "expected_team_b_goals": round(xgb, 3),
            "predicted_total_goals": round(xga + xgb, 3),
            "legacy_base_comparison": None,
            "model_notes": ["Simple Elo converted to xG probabilities (pre-cutoff only)."]
        }
