# World Cup Score Predictor

A statistical football analytics project that predicts the top 5 most probable
scorelines for FIFA World Cup 2026 matches using historical international
results, squad data, and player-level data.

**This is not a betting app.** Output is probabilistic and explainable, not
deterministic.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **No Kaggle credentials needed.**
   The Kaggle player-scores dataset is downloaded via a public HTTP endpoint –
   no `kaggle.json` or API key is required.

## Quick Start

```bash
# 1. Download all raw data (GitHub historical results + Kaggle player-scores)
python -m src.cli download-data

# 2. Parse the squad PDF
python -m src.parse_squads_pdf

# 3. Audit the data
python -m src.cli audit

# 4. Build features (historical + squad + Kaggle player matching)
python -m src.cli build-features

# 5. Predict scorelines (Backtest mode for evaluation)
python -m src.cli backtest

# 6. Evaluate accuracy against actuals
python -m src.cli evaluate

# 7. Predict scorelines (Live mode for upcoming matches)
python -m src.cli predict-live

# 8. Launch the Streamlit dashboard
python -m streamlit run src\app_streamlit.py
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `download-data` | Download GitHub results + Kaggle player-scores |
| `download-kaggle` | Download only Kaggle player-scores (supports `--force`) |
| `audit` | Audit all raw data and produce `data/audit/audit_report.json` |
| `build-features` | Build historical team and squad features |
| `backtest` | Predict scorelines mimicking pre-tournament knowledge (training cutoff 2026-06-10) |
| `evaluate` | Compare backtest predictions against actuals |
| `predict-live` | Predict scorelines using all data up to the current date |
| `simulate-tournament` | Monte Carlo tournament simulation |

## Data Sources

| Source | Method | Credentials |
|--------|--------|-------------|
| [martj42/international_results](https://github.com/martj42/international_results) | Raw CSV download via `requests` | None |
| [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Direct HTTP zip download | **None** |
| `SquadLists-Spanish.pdf` | Local file, parsed with `pdfplumber` | N/A |
| `fixture.csv`, `venues.csv` | Local files | N/A |

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
| Live Prediction Mode | ✅ Implemented |
| Monte Carlo tournament simulation | ⏳ Pending |
