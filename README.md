# World Cup Score Predictor

A statistical football analytics project that predicts the top 5 most probable scorelines for FIFA World Cup 2026 matches using historical international results, squad data, and player-level data.

**This is not a betting app.** Output is probabilistic and explainable, not deterministic.

## Data Sources and Credits

This project relies on several excellent open datasets:

- **[martj42/international_results](https://github.com/martj42/international_results):** Used to calculate historical attack/defense strength via a Poisson model. It is also used during the tournament to pull live actual match results for evaluation and live goal calibration.
- **[davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores):** Used for advanced squad and player strength features (market value, recent appearances, goals, etc.).
- **FIFA SquadLists PDF:** The official source for squad inclusions, caps, and international goals.

**Limitations:** The `player-scores` dataset may lag by several days. However, the model dynamically adjusts to the actual tournament scoring environment by generating a live calibration factor from the most recently updated `martj42` match results.

## Quick Start (Windows PowerShell)

```powershell
# 1. Download all raw data (GitHub historical results + Kaggle player-scores)
python -m src.cli download-data

# 2. Parse the squad PDF
python -m src.parse_squads_pdf

# 3. Audit the data
python -m src.cli audit

# 4. Build features (historical + squad + Kaggle player matching)
python -m src.cli build-features

# 5. Predict scorelines (Default mode)
python -m src.cli predict

# 6. Predict scorelines (Backtest mode for clean evaluation)
python -m src.cli backtest --train-cutoff 2026-06-10

# 7. Evaluate accuracy against actuals (requires actuals from martj42)
python -m src.cli evaluate --actuals-source martj42

# 8. Predict scorelines (Live mode with goal calibration)
python -m src.cli predict-live --as-of-date auto

# 9. Launch the Streamlit dashboard
python -m streamlit run src\app_streamlit.py
```

## Prediction Modes Explained

- **`predict`:** Generates standard baseline predictions without imposing chronological cutoff constraints. Outputs to `predictions.json`.
- **`backtest`:** Mimics pre-tournament knowledge by imposing a strict training cutoff (e.g., `2026-06-10`). It **must not** use any World Cup results after the `train_cutoff`. This ensures an honest, uncontaminated baseline evaluation. Outputs to `backtest_report.json`.
- **`evaluate`:** Compares a prediction report (typically the backtest) against actual recorded match results, outputting hit rates and scoreline concentration diagnostics.
- **`predict-live`:** Predicts upcoming matches using all data up to the current date. It may use already-played World Cup results from the `martj42` dataset to detect if the model is underestimating goal volume, applying an automatic **live calibration factor** to adjust future predictions. Outputs to `live_predictions.json`.

## Implementation Status

| Component | Status |
|-----------|--------|
| GitHub data download | ✅ Implemented |
| Kaggle data download (no credentials) | ✅ Implemented |
| PDF squad parsing | ✅ Implemented |
| Data audit with Kaggle validation | ✅ Implemented |
| Team name normalisation | ✅ Implemented |
| Historical Poisson model | ✅ Implemented |
| Squad feature engineering (Kaggle) | ✅ Implemented |
| Top 5 scoreline predictions | ✅ Implemented |
| Streamlit dashboard | ✅ Implemented |
| Pre-tournament Backtesting | ✅ Implemented |
| Evaluation Module | ✅ Implemented |
| Live Prediction & Calibration | ✅ Implemented |
| Monte Carlo tournament simulation | ⏳ Pending (Not implemented) |
