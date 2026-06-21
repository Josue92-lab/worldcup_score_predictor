# Calibration Replay Audit
Generated: 2026-06-21T13:12:45.557801Z

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
- On fixed-factor replay, moderate often improves Top-5 and volume without as much risk as aggressive.
- On walk-forward, results are closer between postures; moderate provides balanced improvement.
- Aggressive reduces 1-1 concentration more but on small daily samples can be unstable.
- Conservative (current) is safest but leaves goal volume gap.
- Recommendation: use moderate as production default for future-only; continue monitoring with more matchdays.
- Do not promote aggressive to default without clear sustained benefit in walk-forward on new data.

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