from pathlib import Path

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIT_DIR = DATA_DIR / "audit"
OUTPUTS_DIR = DATA_DIR / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Specific File Paths
FIXTURE_PATH = RAW_DIR / "fixture.csv"
VENUES_PATH = RAW_DIR / "venues.csv"
SQUADS_PDF_PATH = RAW_DIR / "SquadLists-Spanish.pdf"

# Kaggle player-scores paths
KAGGLE_DIR = RAW_DIR / "player_scores"
KAGGLE_ZIP_PATH = KAGGLE_DIR / "player-scores.zip"

# Config Files
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yml"
TEAM_ALIASES_PATH = CONFIG_DIR / "team_aliases.yml"

# Output Files
AUDIT_REPORT_PATH = AUDIT_DIR / "audit_report.json"
PREDICTIONS_CSV_PATH = OUTPUTS_DIR / "predictions.csv"
PREDICTIONS_JSON_PATH = OUTPUTS_DIR / "predictions.json"

# Expected Kaggle CSVs (single source of truth)
KAGGLE_EXPECTED_CSVS = [
    "appearances.csv",
    "club_games.csv",
    "clubs.csv",
    "competitions.csv",
    "countries.csv",
    "game_events.csv",
    "game_lineups.csv",
    "games.csv",
    "national_teams.csv",
    "player_valuations.csv",
    "players.csv",
    "transfers.csv",
]

# Create directories if they do not exist
for directory in [RAW_DIR, PROCESSED_DIR, AUDIT_DIR, OUTPUTS_DIR, CONFIG_DIR, KAGGLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
