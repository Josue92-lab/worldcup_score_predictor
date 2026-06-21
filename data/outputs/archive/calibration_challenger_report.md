# Calibration Challenger Report

Generated: 2026-06-21T13:00:36.923750Z

## Overview
This report compares three live calibration postures applied **only to future matches**:
- current: 1.056
- moderate: 1.096
- aggressive: 1.136

Production / default live_predictions.json uses **moderate** posture.

Backtest reports and historical evaluation remain completely uncalibrated (base model).

## Why moderate is the selected production posture
The performance audit showed a raw actual/predicted goal ratio of ~1.148 with 36 evaluated matches (109 actual goals).
The current calibration factor of ~1.056 is a smoothed conservative value (half-step, rolling window 15, bounds).

Moderate (current + ~0.04) steps toward the observed volume gap while remaining within safe bounds (1.10 cap logic) and small-sample caution.

## Why aggressive is not default
Aggressive approaches the raw ratio more closely but risks over-reacting to the current 36-match sample (including possibly anomalous high-scoring games).
We will decide after observing 8-10 additional matches whether to promote aggressive or adjust.

## How to compare scenarios with future results
1. After each matchday, run `python -m src.cli evaluate --actuals-source martj42`
2. Compare the actual goal totals and scorelines against the three live_*.json files (using the future-match predictions that were generated before the matches).
3. Track:
   - Goal volume error per posture
   - Top-1 / Top-5 hit rates per posture (for the matches that were future at generation time)
   - 1-1 concentration shift
   - Whether 1X2 accuracy degrades with higher variance predictions
4. Update production posture in CLI / predict calls when data supports it.

**Important**: Never retro-apply a posture to past matches or backtest reports.

## What to monitor after the next 8-10 matches
- Average actual goals vs the different predicted avgs (current ~2.64, moderate ~2.73?, aggressive ~2.82?)
- Count and % of 1-1 tops in the updated live predictions
- High-scoring matches (5+) frequency and whether they land in top5 more often under higher factors
- Any degradation in aggregate 1X2 accuracy
- Stability of the base calibration computation from calibrate.py

If moderate clearly improves volume without hurting 1X2, consider making it the new "current" base.
Aggressive should only be promoted if data consistently shows underestimation beyond what moderate corrects.
