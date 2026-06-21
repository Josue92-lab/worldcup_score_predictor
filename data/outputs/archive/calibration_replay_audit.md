# Calibration Replay Audit
Generated: 2026-06-21T13:21:45.106276Z

## Overview
Retrospective calibration replay on the 36 evaluated matches (2026-06-11 to 2026-06-20).
Base predictions come from backtest run with train cutoff 2026-06-10 (no World Cup results used).

**IMPORTANT DISTINCTION:**
- FIXED-FACTOR REPLAY IS RETROSPECTIVE SENSITIVITY TEST. The factors (including moderate/aggressive chosen with full knowledge of results) are applied to the known sample.
- WALK-FORWARD REPLAY IS CLOSER TO HONEST FUTURE-ONLY TEST. Factor for each day is derived using only results up to the previous day.

**Two modes:**
- Fixed-factor replay: applies the three static factors to all base predictions. Retrospective sensitivity test.
- Walk-forward replay: for each matchday, derives the factor using only previously observed results, then evaluates that day.

## Postures
- conservative: factor=1.056 - Current or near-current live factor
- moderate: factor=1.096 - Moderately higher goal-volume inflation
- aggressive: factor=1.136 - Stronger goal-volume inflation, shadow only

## Fixed-Factor Replay Results
### base (factor 2.638 avg pred)
- Top-1: 0.1944, Top-5: 0.5, 1X2: 0.4722
- Goal gap: 0.39, 1-1%: 94.4%
- High scoring (>4) misses: 8
### conservative (factor 2.785 avg pred)
- Top-1: 0.1944, Top-5: 0.4444, 1X2: 0.4722
- Goal gap: 0.243, 1-1%: 94.4%
- High scoring (>4) misses: 8
### moderate (factor 2.891 avg pred)
- Top-1: 0.1944, Top-5: 0.4167, 1X2: 0.4722
- Goal gap: 0.137, 1-1%: 94.4%
- High scoring (>4) misses: 8
### aggressive (factor 2.996 avg pred)
- Top-1: 0.1944, Top-5: 0.4444, 1X2: 0.4722
- Goal gap: 0.032, 1-1%: 91.7%
- High scoring (>4) misses: 8

## Walk-Forward Replay Results
### base
- Top-1: 0.1944, Top-5: 0.5, 1X2: 0.4722
- Goal gap: 0.39, 1-1%: 94.4%
### conservative
- Top-1: 0.1944, Top-5: 0.4444, 1X2: 0.4722
- Goal gap: 0.314, 1-1%: 94.4%
### moderate
- Top-1: 0.1944, Top-5: 0.4167, 1X2: 0.4722
- Goal gap: 0.219, 1-1%: 94.4%
### aggressive
- Top-1: 0.1944, Top-5: 0.4167, 1X2: 0.4722
- Goal gap: 0.114, 1-1%: 94.4%

## Key Comparisons (from summary)
Fixed best Top-5: conservative
Fixed best 1X2: conservative
Walk-forward best Top-5: conservative
Walk-forward best 1X2: conservative

## Recommendation
Recommended posture: **moderate**
- Base (factor 1.0) has the best Top-5 rate in this replay.
- Among calibrated postures, conservative performs at least as well as moderate/aggressive on Top-5 and 1X2.
- 1X2 rates are essentially identical across all postures in both replay modes.
- Moderate (and higher) mainly improves the goal-volume gap (reduces underestimation).
- moderate is recommended ONLY if the priority is goal-volume calibration for future matches, not because it improves Top-5 or 1X2 accuracy on this replay sample.
- Aggressive does not clearly outperform on accuracy metrics and increases risk of over-correction on small sample.
- More data needed; this replay does not strongly support changing the production default away from conservative if accuracy is primary.

## Warnings
- FIXED-FACTOR REPLAY IS RETROSPECTIVE and uses factors informed by full sample. Do not interpret as prospective performance.
- WALK-FORWARD uses only past data for calibration decisions per day - more valid for testing policy.
- Small sample (36 matches, 10 days). Conclusions should be tentative.
- High-scoring games (5+) heavily influence volume metrics.
- This experiment does not change any production predictions or backtest.

## Notes on Interpretation
Fixed-factor results can appear better because the factors were selected knowing the full sample outcomes.
Walk-forward is the relevant test for whether a policy would have worked prospectively.
More matchdays needed before promoting any posture change to production.

## Sensitivity Grid (fixed-factor, analysis only)
Factor | Avg Pred Goals | Gap | Top5% | 1X2% | 1-1% | Top5 changed | 1X2 changed
-------|----------------|-----|-------|------|------|--------------|-------------
1.00 | 2.638 | 0.39 | 50.0% | 47.2% | 94.4% | 0 | 0
1.05 | 2.77 | 0.258 | 44.4% | 47.2% | 94.4% | 26 | 0
1.10 | 2.902 | 0.126 | 41.7% | 47.2% | 94.4% | 35 | 0
1.15 | 3.033 | -0.005 | 44.4% | 47.2% | 91.7% | 36 | 0
1.20 | 3.165 | -0.137 | 44.4% | 47.2% | 86.1% | 36 | 0
1.25 | 3.297 | -0.269 | 41.7% | 47.2% | 80.6% | 36 | 0
1.30 | 3.429 | -0.401 | 44.4% | 47.2% | 80.6% | 36 | 0

## Model-Family Sensitivity Notes
A global lambda (goal-volume) multiplier primarily scales total expected goals but preserves the relative attack/defense strengths between teams.
As a result:
- It can improve average goal volume match (reduce the +0.39 gap).
- It often has little or no effect on aggregate 1X2 outcome probabilities.
- It may reduce 1-1 concentration only modestly, because the mode of the distribution remains similar relative to team strength difference.
De-concentrating 1-1 or meaningfully improving 1X2 may require other adjustments such as:
  - asymmetric favorite/underdog scaling,
  - draw dampening,
  - sensitivity on Dixon-Coles rho,
  - or better separation in underlying team strength estimates.
These are outside the scope of pure lambda calibration and were not implemented in this replay.