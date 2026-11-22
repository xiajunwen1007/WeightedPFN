"""WeightedPFN: a tabular foundation model for weighted regression and weighted classification (scikit-learn interface).

WeightedPFN is TabPFN-3.5 with weighted attention, fine-tuned for 800 steps on synthetic missing-data tasks.
"""
from ._patch import process_weights
from .estimators import (
    DEFAULT_CLASSIFIER_WEIGHTS,
    DEFAULT_REGRESSOR_WEIGHTS,
    WeightedPFNClassifier,
    WeightedPFNRegressor,
    clear_model_cache,
    resolve_device,
)
from .hub import download_weights

__version__ = "1.0.0"
__all__ = [
    "WeightedPFNClassifier",
    "WeightedPFNRegressor",
    "DEFAULT_CLASSIFIER_WEIGHTS",
    "DEFAULT_REGRESSOR_WEIGHTS",
    "clear_model_cache",
    "download_weights",
    "process_weights",
    "resolve_device",
]
