# World Cup Score Predictor

**A reproducible, auditable, educational probability model for World Cup scoreline and outcome forecasting.**

## Overview

This app generates probabilistic predictions for international football matches (focusing on the 2026 World Cup) and exposes the models transparently for inspection and backtesting.

- **Exact scoreline predictions**: Top-5 most probable final scores per match.
- **Aggregate 1X2 probabilities**: Win / Draw / Win for the two teams.
- **Goal volume diagnostics**: Expected total goals and calibration signals.
- **Model comparison**: Live mode uses the best available engine; legacy base model remains available for benchmarking.

The system is designed for learning about probabilistic modeling, not for betting.

## Core v2 (Current Live Default)

**Core v2 Hybrid** is the default engine for live predictions:

- **Scoreline engine**: `ensemble` (best Top-5 exact scoreline coverage in bake-off testing).
- **1X2 engine**: `hybrid_elo_poisson` (strongest aggregate outcome probabilities on Brier score and log loss).
- **Goal-volume policy**: Moderate calibration (lambda ≈ 1.15 style).
- **Transparency**: Includes legacy base comparison and engine divergence diagnostics.

Core v2 is the best available live engine based on current bake-off evidence, but it remains an educational/prototype forecasting model.

## Legacy Base Model

The original Poisson + Dixon-Coles baseline is preserved as:

- Honest backtest reference.
- Educational benchmark.
- Fallback option.

Backtest mode defaults to the legacy base for historical honesty.

## Model Architecture

| Model                | Purpose                              | Key Strength                     | Notes |
|----------------------|--------------------------------------|----------------------------------|-------|
| `base`               | Legacy Poisson/Dixon-Coles baseline | Transparent, simple              | Benchmark & backtest default |
| `lambda_1.15`        | Goal-volume calibrated Poisson       | Matches observed goal totals     | Improves volume gap |
| `hybrid_elo_poisson` | Elo strength blended with Poisson    | Best 1X2 probabilistic metrics   | Strong aggregate outcomes |
| `ensemble`           | Multi-engine scoreline combination   | Best Top-5 exact score coverage  | Good for exact score focus |
| `core_v2` (default)  | Hybrid production engine             | Balanced across metrics          | Live default; ensemble + hybrid + moderate calibration |

Global lambda calibration alone improves volume but does not solve discrimination or 1-1 concentration issues. Core v2 combines the strongest tested components.

## Prediction Modes

- **Live** (`predict-live --as-of-date auto`): Default = Core v2. May use results up to the as-of date for calibration of *future* matches only. Not a backtest.
- **Backtest** (`backtest --train-cutoff YYYY-MM-DD`): Strict cutoff. No post-cutoff data leaks into parameters or calibration. Honest evaluation of skill. Supports `--model base` or `--model core_v2`.
- **Evaluate**: Matches predictions against actual results (Top-1, Top-5, 1X2, goal volume).
- **Model Bake-off**: Runs full comparison of model families using only pre-cutoff data.

**Critical honesty rule**: Backtest results are never retroactively calibrated with future match results.

## Data Sources

- Historical international results & live actuals: [martj42/international_results](https://github.com/martj42/international_results)
- Player valuations, appearances, club form: [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- Squad lists: FIFA SquadLists PDF (parsed)
- Fixtures and venues: `data/raw/fixture.csv` + `venues.csv`

Player/squad data is used primarily for data quality flags and explanations. Core numeric forecasts come from historical team strengths + calibration.

## Quick Start (PowerShell)

```powershell
# Download raw data
python -m src.cli download-data

# (Optional) Parse squad PDF and audit
python -m src.parse_squads_pdf
python -m src.cli audit

# Build features
python -m src.cli build-features

# Honest backtest (legacy base - recommended for claims)
python -m src.cli backtest --train-cutoff 2026-06-10 --model base

# Backtest with Core v2 for comparison
python -m src.cli backtest --train-cutoff 2026-06-10 --model core_v2

# Evaluate against actuals
python -m src.cli evaluate --actuals-source martj42

# Live predictions (Core v2 default)
python -m src.cli predict-live --as-of-date auto --model core_v2

# Full model bake-off audit
python -m src.cli model-bakeoff

# Launch dashboard
python -m streamlit run src\app_streamlit.py
```

## CLI Commands

- `python -m pytest` — Run tests.
- `python -m src.cli backtest --train-cutoff 2026-06-10 --model base` — Legacy honest backtest.
- `python -m src.cli backtest --train-cutoff 2026-06-10 --model core_v2` — Core v2 backtest.
- `python -m src.cli evaluate --actuals-source martj42` — Match predictions to actual results.
- `python -m src.cli predict-live --as-of-date auto --model core_v2` — Generate live predictions (Core v2).
- `python -m src.cli model-bakeoff` — Run full model family comparison.
- `python -m streamlit run src\app_streamlit.py` — Interactive dashboard.

## Streamlit Dashboard

The dashboard defaults to Core v2 predictions when available.

- Model selector allows switching between **Core v2 Hybrid** and **Legacy Base**.
- Core v2 banner explains the engines in use.
- Diagnostics panel shows hit rates, goal averages, 1-1 concentration, and calibration signals.
- Legacy base remains available for side-by-side comparison.

## Output Files

Important generated artifacts (in `data/outputs/`):

- `live_predictions.json` — Default live (Core v2)
- `live_predictions_core_v2.json` — Explicit Core v2 output
- `live_predictions_legacy_base.json` — Legacy base for comparison
- `backtest_report.json` — Honest pre-tournament predictions
- `evaluation_report.json` — Actuals matching + metrics
- `core_v2_comparison_report.json` / `.md` — Core v2 vs alternatives
- `model_bakeoff_audit.json` / `.md` / `leaderboard.csv` — Full bake-off results

## Evaluation and Backtesting

- **Train cutoff** enforces no leakage: features and calibration for a backtest use only data up to the cutoff.
- Live calibration is **future-only**.
- Evaluation uses matched actual results from the martj42 feed.
- Key metrics: Top-1/Top-5 exact hit rates, 1X2 accuracy, Brier score, log loss, goal gap, 1-1 concentration.

## Model Bake-off Summary

Testing on the 2026-06-11 → 2026-06-20 window (pre-cutoff models only) showed:

- Base model exhibited high 1-1 concentration and modest 1X2 skill.
- Global lambda calibration improved goal-volume match but delivered limited gains on Top-5 or 1X2.
- Ensemble improved exact scoreline coverage (Top-5).
- Hybrid Elo-Poisson delivered the strongest 1X2 probabilistic scoring (Brier / log loss).
- Core v2 combines these strengths for the current live default.

The evaluation sample is small; results should be treated as directional evidence requiring ongoing monitoring.

## Model Risk and Limitations

**Educational / prototype status**: Amber  
**High-stakes forecasting readiness**: Red

Limitations:

- Small evaluation sample (tens of matches).
- Exact score prediction remains inherently difficult.
- Historical data quality and coverage vary.
- No current injuries, lineups, or late news.
- Global factors and simple hybrids have limits.

This is **not** a betting tool. Use it to understand uncertainty and model behavior.

Monitor after each matchday. Future validation on new data is required before stronger claims.

## Repository Structure

```
src/
  models/
    base_poisson.py
    lambda_calibrated.py
    elo.py
    hybrid_elo_poisson.py
    ensemble.py
    core_v2.py
    registry.py
  predict_scorelines.py
  evaluate_predictions.py
  calibrate.py
  cli.py
  app_streamlit.py
  ...
data/
  outputs/
    live_predictions*.json
    backtest_report.json
    model_bakeoff_* / core_v2_*
  raw/
  processed/
```

## Development Notes

When adding models:

1. Implement in `src/models/`.
2. Register in `registry.py`.
3. Add entry to bake-off (pre-cutoff only).
4. Update model cards with strengths/weaknesses.
5. Ensure `--model` support in CLI and no leakage.
6. Add tests for the new engine.
7. Document in README and comparison reports.

All model selection / calibration must be done on pre-cutoff validation data.

## Data Credits

See the in-app sidebar and the credits section above.

## License

Educational use. See individual dataset licenses.

---

**This README reflects Core v2 as the live default and the legacy base as the benchmark/fallback.** All historical backtests remain honest.
