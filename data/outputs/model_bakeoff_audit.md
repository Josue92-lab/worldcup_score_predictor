# World Cup Score Predictor - Full Model Bake-off Audit

Generated: 2026-06-23T21:06:34.435646Z
Train cutoff: 2026-06-10
Test matches: 44

## Executive Verdict
**Recommended action: use_moderate_calibration (model: lambda_1.15)**
Current Poisson + Dixon-Coles base is structurally limited on this sample: extreme 1-1 concentration (~94% in base) and flat ~47% 1X2.
Global lambda helps volume but does little for Top-5 or 1X2.
Hybrid/recency + targeted dampening or Elo hybrids show better balance on proper scoring rules.
Production readiness: amber (educational) / red (high-stakes forecasting).

## Leaderboard (balanced sort)
1. lambda_1.15: Top5=0.4091, Brier=0.6329, Gap=0.026, 1-1%=90.9
2. lambda_1.20: Top5=0.4091, Brier=0.6336, Gap=-0.105, 1-1%=84.1
3. lambda_1.10: Top5=0.3636, Brier=0.6323, Gap=0.157, 1-1%=95.5
4. lambda_1.25: Top5=0.4091, Brier=0.6345, Gap=-0.237, 1-1%=79.5
5. elo: Top5=0.4545, Brier=0.5602, Gap=0.367, 1-1%=40.9
6. hybrid_elo_poisson: Top5=0.4545, Brier=0.5557, Gap=0.393, 1-1%=81.8
7. lambda_1.05: Top5=0.3864, Brier=0.6318, Gap=0.288, 1-1%=95.5
8. asym_fav: Top5=0.4318, Brier=0.6345, Gap=0.337, 1-1%=72.7

## Key Findings
- Best Top-5: {'n': 44, 'avg_actual_goals': 3.045, 'avg_predicted_goals': 2.631, 'goal_gap': 0.414, 'ratio': 1.157, 'top1_rate': 0.1591, 'top5_rate': 0.5, 'o1x2_rate': 0.5682, 'brier_1x2': 0.6038, 'logloss_1x2': 1.0044, 'exact_nll_approx': 4.1831, 'one_one_pct': 95.5, 'high_miss_gt4': 9, 'avg_top_p': 0.1355, 'avg_top5_mass': 0.5001}
- 1X2 largely insensitive to global lambda.
- 1-1 concentration primarily from low total goals + DC adjustment + modest team separation.
- Ensemble and targeted dampening reduce 1-1 while preserving or improving volume capture.

## Model Risk Ratings (Green/Amber/Red)
- Leakage control: Green (strict cutoff enforced)
- Calibration (1X2 + volume): Amber
- Exact score quality: Red (base)
- Production readiness: Amber

## Recommendation Details
- Current base is conservative and 1-1 heavy.
- Global lambda (1.10-1.15) dramatically improves volume with little harm to Top-5/1X2 on this sample.
- Targeted draw dampening and asymmetric fav scaling reduce 1-1 concentration better than volume alone.
- Hybrid/Elo show promise but small n=36 means no model family wins decisively across all metrics.
- Safest: retain current base for backtest honesty; apply moderate calibration (around 1.10-1.15) as future-only challenger. Do not switch core model family yet.

## Remaining Issues
- Limited sample size
- Simple Elo implementation
- No full pre-cutoff cross-validation loop in this run

See JSON for full per-match, ablation, and metrics.