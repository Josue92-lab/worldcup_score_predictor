"""
WorldCup Predictor Core v2 - Model package.

Provides versioned model engines for scorelines, 1X2, and calibration.
"""

from .registry import get_model, list_models, predict_match
from .core_v2 import CoreV2Predictor
from .base_poisson import BasePoissonModel
from .ensemble import EnsembleModel
from .hybrid_elo_poisson import HybridEloPoissonModel
from .lambda_calibrated import LambdaCalibratedModel

__all__ = [
    "get_model",
    "list_models",
    "predict_match",
    "CoreV2Predictor",
    "BasePoissonModel",
    "EnsembleModel",
    "HybridEloPoissonModel",
    "LambdaCalibratedModel",
]
