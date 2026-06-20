# World Cup Score Predictor

**A reproducible, auditable, educational probability model for World Cup scoreline scenarios.**

## What this app is

- An interactive dashboard that shows the **top 5 most probable exact scorelines** for each fixture using a transparent Poisson model (with Dixon-Coles low-score adjustment).
- Designed to help users understand probabilistic forecasting: why the single most likely score can differ from the most likely 1X2 outcome, how models can be backtested, and what calibration does.
- Exposes its own performance (Top-1 / Top-5 / 1X2 hit rates), data quality signals, and limitations.

## What it is not

**Not a betting tool. Not a guarantee of results. Not an oracle.** Exact score prediction is inherently uncertain. The app deliberately shows when its predictions miss.

## Data source credits

- **Historical international results & live actuals for evaluation/calibration**: [martj42/international_results](https://github.com/martj42/international_results)
- **Player valuations, club form, appearances for squad context**: [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- **Official squad lists**: FIFA SquadLists PDF (48 teams × 26 players parsed)
- **Fixtures and venue/timezone metadata**: `data/raw/fixture.csv` + `venues.csv` (dates corrected to local venue time where needed)

**Dataset freshness**: Player-level data (Kaggle) may lag by days. Match results from martj42 are refreshed independently. Low coverage on some squads is flagged in the UI.

## How the model works (high level)

1. Historical international matches (pre-cutoff in backtest mode) → empirical **attack strength** and **defense strength** per team (relative to average).
2. **Poisson** distribution produces a full scoreline probability matrix for expected goals of Team A and Team B.
3. Optional **Dixon-Coles** adjustment (rho ≈ -0.13) increases probability mass on low-score outcomes (0-0, 1-0, 0-1, 1-1).
4. Top 5 scorelines are taken directly from the normalized probability matrix.
5. **1X2 probabilities** (Team A win / draw / Team B win) are *aggregates* — the sum of probabilities of every scoreline that produces that outcome. This is why `1-1` can be the #1 exact score while one team still has >50% chance to win overall.

**Important honesty about player / squad data**: Player and squad features (market value, SQI, recent club stats) are currently used **only** for:
- Data quality warnings
- The "Explanation & Data Quality" panel
- Generating human-readable "main factors"

The numerical forecast (xG / scoreline probabilities) is driven by **historical team attack/defense** + the live goal-volume calibration factor. Squad data does not directly adjust the lambdas in the current implementation.

## Backtest vs live prediction modes

- **Backtest** (`python -m src.cli backtest --train-cutoff 2026-06-10`): Uses a strict training cutoff. Historical features are filtered to data **before** the cutoff. No World Cup 2026 results after the cutoff are used for fitting or calibration. Intended for honest model evaluation.
- **Live** (`python -m src.cli predict-live --as-of-date auto`): May incorporate already-played World Cup results up to the `as_of_date` to compute a rolling calibration factor (fav/underdog asymmetric) that inflates or deflates future goal expectations. **Live calibration must not be used retroactively to claim past predictive skill.**

The Streamlit app lets you load either `backtest_report.json` or `live_predictions.json` (plus the evaluation/calibration artifacts).

## Prediction interpretation (key concepts the UI teaches)

- **Top Prediction** = single most probable exact final score (e.g. 1-1).
- **Team A / Team B Win %** + Draw % = aggregated probability across dozens of scorelines.
- **Top 5 Hit** = actual result landed in the model's 5 most likely exact scores.
- **1X2 Hit** = model picked the correct broad category by argmax of the three *aggregate* probabilities (independent of whether the exact score or top-5 matched).
- **Pending** = result not yet available in the results feed.

See the in-app "How to read these predictions" panel for the full plain-language guide.

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

## Prediction Modes Explained (see also the UI banners)

- **`predict`:** Standard run. No enforced cutoff. Useful for development.
- **`backtest --train-cutoff YYYY-MM-DD`:** The mode you should use for any claim about model skill. Training data strictly ends at the cutoff. No post-cutoff World Cup results leak into attack/defense parameters or calibration.
- **`evaluate`:** Matches predictions (usually backtest) against actual results from the martj42 feed. Computes Top-1, Top-5 and 1X2 hit rates + goal volume diagnostics.
- **`predict-live --as-of-date auto|YYYY-MM-DD`:** Uses results up to the date for calibration only. For "what would we predict today". Do **not** treat the numbers as a backtest of past accuracy.

**Live calibration is for operational use only.** It should never be turned on retroactively and then used to publish "our backtest hit rate improved".

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

## Trust & limitations (displayed in app)

This app is a reproducible educational probability model, not a betting tool or oracle. Predictions are generated from historical international match rates using a Poisson model with optional Dixon-Coles adjustment and live goal-volume calibration. The model exposes its own performance through backtesting and actual-result comparison.

It does not know current injuries, confirmed starting lineups, tactical plans, weather, motivation, or last-minute news. Player and squad data may lag behind real-world changes. Use the app to understand probability, uncertainty, and model calibration — not as a guarantee of match results.

**Player/squad data note**: Currently used for context and data quality flags only. The core scoreline probabilities come from historical team strengths + calibration.

## In-app educational features (as of this update)

- "How to read these predictions" panel (top of Dashboard)
- Mode-specific banners (Backtest vs Live)
- "Trust and limitations" guidance (sidebar + README)
- Readable Explanation & Data Quality bullets (with raw JSON toggle)
- Model diagnostics panel (hit rates, goal averages, calibration factor, 1-1 concentration warnings)
- Debug information hidden behind sidebar checkbox
- Data credits in sidebar + footer

