from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.audit_data import is_knockout_placeholder

def test_knockout_placeholder_detection():
    assert is_knockout_placeholder("Winner Group A") is True
    assert is_knockout_placeholder("Runner-up Group B") is True
    assert is_knockout_placeholder("Third place Group A/B/C") is True
    assert is_knockout_placeholder("Mexico") is False
    assert is_knockout_placeholder("USA") is False
    assert is_knockout_placeholder("Winner Match 73") is True
