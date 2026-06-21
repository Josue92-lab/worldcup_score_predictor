# World Cup Score Predictor - Performance Audit
Generated: 2026-06-21T12:51:05.807665Z

## Executive Verdict
The model is systematically underestimating goal volume. It is overly concentrated on 1-1 (94% of top predictions). Exact score accuracy is low (19% top-1, 50% top-5) while aggregate 1X2 is ~47% (near-random). Live calibration is active but conservative.

## Current Accuracy
- Evaluated matches: 36 (pending 36)
- Top-1 exact: 19.4% (7/36)
- Top-5 exact: 50.0% (18/36)
- 1X2 aggregate: 47.2% (17/36)
- Brier (1X2 approx): 0.6341

## Goal-Volume Finding
- Avg actual goals: 3.028 vs predicted 2.638 (gap +0.39)
- Actual / predicted ratio: 1.148
- 8 matches with 5+ actual goals had 8 high-score misses (code threshold actual_goals > 4). 6 matches had exactly 4 goals.
- Model is underestimating goal volume with medium confidence.

## 1-1 Concentration Finding
- 1-1 is top prediction in 34/36 (94.4%)
- This is extreme. Indicates baseline is too conservative/dispersed too little.

## Best Model Signals
- Top-5 captures half the actuals.
- Some signal in aggregate outcome, but weak.

## Worst Failure Modes
- High scoring games completely missed in top 5.
- Heavy bias toward 1-1 on almost all matches.
- 1X2 barely better than chance.

## Should live calibration be more aggressive?
**Yes, moderately.**
Current factor: 1.056. Raw data suggests ~1.148.
Recommendation: move toward ~1.09-1.10 range for future matches only.
Monitor after next 8-10 results before going further.

## What to monitor after next matchday
- Actual total goals vs 2.64-2.8 baseline.
- % of new top predictions that are 1-1.
- High score games frequency.
- Whether moderate inflation hurts 1X2 or helps overall calibration.