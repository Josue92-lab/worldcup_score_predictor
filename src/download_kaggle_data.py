"""
Download Kaggle player-scores dataset via direct HTTP (no credentials needed).

This dataset is publicly accessible at:
  https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores

The zip contains:
  appearances.csv, club_games.csv, clubs.csv, competitions.csv,
  countries.csv, game_events.csv, game_lineups.csv, games.csv,
  national_teams.csv, player_valuations.csv, players.csv, transfers.csv
"""

import sys
import zipfile
import requests
from pathlib import Path
from src.config import RAW_DIR

KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores"
)

EXPECTED_CSVS = [
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

TARGET_DIR = RAW_DIR / "player_scores"
ZIP_PATH = TARGET_DIR / "player-scores.zip"


def _all_csvs_present() -> bool:
    """Return True only when the zip AND every expected CSV exist."""
    if not ZIP_PATH.exists():
        return False
    for csv in EXPECTED_CSVS:
        if not (TARGET_DIR / csv).exists():
            return False
    return True


def download_kaggle_data(force: bool = False) -> bool:
    """
    Download, extract, and validate the Kaggle player-scores dataset.

    Parameters
    ----------
    force : bool
        If True, re-download even when all files already exist.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # ── idempotency ──────────────────────────────────────────────────────
    if not force and _all_csvs_present():
        print(f"[kaggle] All expected CSV files already present in {TARGET_DIR}")
        print("[kaggle] Skipping download.  Use --force to re-download.")
        return True

    # ── download ─────────────────────────────────────────────────────────
    print(f"[kaggle] Downloading from {KAGGLE_DOWNLOAD_URL} ...")
    response = None
    try:
        response = requests.get(
            KAGGLE_DOWNLOAD_URL, stream=True, timeout=(30, 300)
        )
        response.raise_for_status()
    except requests.exceptions.SSLError:
        # Corporate proxies often inject their own CA; retry without verification
        print("[kaggle] WARNING: SSL certificate verification failed.")
        print("[kaggle]   Retrying with verify=False (corporate proxy detected).")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(
                KAGGLE_DOWNLOAD_URL, stream=True, timeout=(30, 300), verify=False
            )
            response.raise_for_status()
        except requests.RequestException as exc2:
            print(f"[kaggle] ERROR: HTTP request failed even without SSL verify: {exc2}")
            return False
    except requests.RequestException as exc:
        print(f"[kaggle] ERROR: HTTP request failed: {exc}")
        return False

    # ── stream to disk with progress ─────────────────────────────────────
    content_length = response.headers.get("Content-Length")
    total_bytes = int(content_length) if content_length else None

    downloaded = 0
    with open(ZIP_PATH, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / (1024 * 1024)
                if total_bytes:
                    pct = downloaded / total_bytes * 100
                    print(
                        f"\r[kaggle]   {mb:,.1f} MB / {total_bytes / (1024*1024):,.1f} MB  ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r[kaggle]   {mb:,.1f} MB downloaded", end="", flush=True)
    print()  # newline after progress

    file_size = ZIP_PATH.stat().st_size
    print(f"[kaggle] Saved zip to {ZIP_PATH}  ({file_size / (1024*1024):,.1f} MB)")

    # ── validate it is really a zip ──────────────────────────────────────
    if not zipfile.is_zipfile(ZIP_PATH):
        print(
            "[kaggle] ERROR: Downloaded file is not a valid zip. "
            "Kaggle may have changed access requirements or the request was blocked."
        )
        # Remove the bad file so next run retries
        ZIP_PATH.unlink(missing_ok=True)
        return False

    # ── extract ──────────────────────────────────────────────────────────
    print(f"[kaggle] Extracting to {TARGET_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(TARGET_DIR)
    print("[kaggle] Extraction complete.")

    # ── validate expected CSVs ───────────────────────────────────────────
    missing = [csv for csv in EXPECTED_CSVS if not (TARGET_DIR / csv).exists()]
    if missing:
        print(f"[kaggle] WARNING: {len(missing)} expected CSV(s) missing after extraction:")
        for m in missing:
            print(f"  - {m}")
        return False

    print(f"[kaggle] All {len(EXPECTED_CSVS)} expected CSV files present.")
    return True


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    success = download_kaggle_data(force=force_flag)
    if not success:
        sys.exit(1)
