# World Cup Score Predictor - Full Model Bake-off Audit

Generated: 2026-06-21T14:06:51.296583Z
Train cutoff: 2026-06-10
Test matches: 36

## Executive Verdict
**Recommended action: use_moderate_calibration (model: lambda_1.15)**
Current Poisson + Dixon-Coles base is structurally limited on this sample: extreme 1-1 concentration (~94% in base) and flat ~47% 1X2.
Global lambda helps volume but does little for Top-5 or 1X2.
Hybrid/recency + targeted dampening or Elo hybrids show better balance on proper scoring rules.
Production readiness: amber (educational) / red (high-stakes forecasting).

## Leaderboard (balanced sort)
1. lambda_1.15: Top5=0.4444, Brier=0.636, Gap=-0.005, 1-1%=91.7
2. lambda_1.20: Top5=0.4444, Brier=0.637, Gap=-0.137, 1-1%=86.1
3. lambda_1.10: Top5=0.4167, Brier=0.6352, Gap=0.126, 1-1%=94.4
4. lambda_1.05: Top5=0.4444, Brier=0.6345, Gap=0.258, 1-1%=94.4
5. lambda_1.25: Top5=0.4167, Brier=0.6381, Gap=-0.269, 1-1%=80.6
6. hybrid_elo_poisson: Top5=0.4722, Brier=0.5717, Gap=0.374, 1-1%=83.3
7. asym_fav: Top5=0.4722, Brier=0.6354, Gap=0.311, 1-1%=77.8
8. ensemble: Top5=0.5556, Brier=0.6098, Gap=0.387, 1-1%=94.4

## Key Findings
- Best Top-5: {'n': 36, 'avg_actual_goals': 3.028, 'avg_predicted_goals': 2.641, 'goal_gap': 0.387, 'ratio': 1.147, 'top1_rate': 0.1944, 'top5_rate': 0.5556, 'o1x2_rate': 0.5556, 'brier_1x2': 0.6098, 'logloss_1x2': 1.0132, 'exact_nll_approx': 4.0893, 'one_one_pct': 94.4, 'high_miss_gt4': 8, 'avg_top_p': 0.1358, 'avg_top5_mass': 0.4968}
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