import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from unittest import mock
import pytest
from src.cli import main
from src.config import PREDICTIONS_JSON_PATH

@mock.patch("src.predict_scorelines.predict_scorelines")
def test_cli_predict_calls_correct_mode(mock_predict_scorelines, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli", "predict"])
    main()
    mock_predict_scorelines.assert_called_once_with(mode="predict")
