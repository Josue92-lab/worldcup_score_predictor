"""
Model Registry for WorldCup Predictor Core v2.

Allows selecting different engines for different prediction components.
Strictly respects train_cutoff for honesty.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path
import sys
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR
from .base_poisson import BasePoissonModel
from .ensemble import EnsembleModel
from .hybrid_elo_poisson import HybridEloPoissonModel
from .lambda_calibrated import LambdaCalibratedModel

# Registry of available models
MODELS: Dict[str, Any] = {}

def register_model(name: str, model_class: Any, description: str = ""):
    MODELS[name] = {"class": model_class, "description": description}

def get_model(name: str = "core_v2", **kwargs) -> Any:
    """Get instantiated model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    entry = MODELS[name]
    return entry["class"](**kwargs)

def list_models() -> Dict[str, str]:
    return {name: entry["description"] for name, entry in MODELS.items()}

def predict_match(match: dict, context: dict = None, model_name: str = "core_v2", **kwargs) -> dict:
    """
    Unified prediction interface.
    match: dict with team_a, team_b, date, etc. from fixture.
    context: optional with historical data, cutoff, etc.
    """
    model = get_model(model_name, **kwargs)
    return model.predict(match, context)

# Register standard models
register_model("base", BasePoissonModel, "Legacy Poisson/Dixon-Coles baseline (honest backtest reference)")
register_model("lambda_1.15", LambdaCalibratedModel, "Goal-volume calibrated Poisson (factor ~1.15)")
register_model("hybrid_elo_poisson", HybridEloPoissonModel, "Best 1X2 probabilistic engine from bake-off")
register_model("ensemble", EnsembleModel, "Best scoreline coverage engine from bake-off")
register_model("core_v2", lambda **kw: CoreV2Predictor(**kw), "Production hybrid: ensemble scoreline + hybrid 1X2 + moderate calibration")

# Note: core_v2 is instantiated specially below in core_v2.py import to avoid circularity
# We will override after import

from .core_v2 import CoreV2Predictor
MODELS["core_v2"]["class"] = CoreV2Predictor
