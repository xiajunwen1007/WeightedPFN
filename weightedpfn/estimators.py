"""scikit-learn estimators for WeightedPFN (CPU or GPU, FP32 inference).

``fit(X, y, sample_weight)`` stores the weighted context (no network pass, as TabPFN's ``fit_preprocessors`` mode);
``predict`` / ``predict_proba`` run the network with the weighted attention and weighted column preprocessing.
The settings below are the ones used in the experiments of the paper: one estimator, FP32, seed 2026101202,
query chunks of 16384 rows; the regressor uses no target transform and no row subsampling.
"""
from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from . import _patch

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
# tabpfn reads the model version from "v3.5" in the file name (it sets the CPU row limit): keep it in the names
DEFAULT_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "weightedpfn-v3.5-classifier.safetensors"
DEFAULT_REGRESSOR_WEIGHTS = WEIGHTS_DIR / "weightedpfn-v3.5-regressor.safetensors"
SEED = 2026101202
QUERY_CHUNK = 16384
REGRESSION_INFERENCE_CONFIG = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,), "SUBSAMPLE_SAMPLES": None}

_NETWORKS = {}  # one network per (task, checkpoint file, device), shared by all estimators of this process
_NETWORK_LOCK = threading.Lock()


def resolve_device(device="auto"):
    """'auto' -> 'cuda' when a GPU is visible, else 'cpu'; also accepts 'cpu', 'cuda', 'cuda:1', torch.device."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={str(device)!r} was requested but no CUDA GPU is available; use device='cpu'.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or a CUDA device such as 'cuda' or 'cuda:0'.")
    return str(device)


def clear_model_cache():
    """Release the networks kept in memory by this process."""
    with _NETWORK_LOCK:
        _NETWORKS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _network(kind, model_path, options):
    """Build the network from a TabPFN-3.5 safetensors file once and hand it to TabPFN as in-memory ModelSpecs."""
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.base import ModelSpecs

    from tabpfn.architectures.tabpfn_v3_5 import TabPFNV3p5

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}. Download them with "
                                "weightedpfn.download_weights() and pass model_path=..., or point "
                                "model_path to a local copy.")
    if "v3.5" not in path.name:
        raise ValueError(f"{path.name}: the file name must contain 'v3.5' (tabpfn reads the model version from it).")
    stat = path.stat()
    key = (kind, str(path), stat.st_mtime_ns, stat.st_size, options["device"])
    with _NETWORK_LOCK:
        if key not in _NETWORKS:
            estimator = (TabPFNClassifier if kind == "classifier" else TabPFNRegressor)(**dict(options, model_path=str(path)))
            estimator.get_inference_config()  # loads the checkpoint
            model = estimator.models_[0].eval().requires_grad_(False)
            if not isinstance(model, TabPFNV3p5):
                raise ValueError(f"{path.name} is not a TabPFN-3.5 network; the weighting patch supports TabPFN-3.5 only.")
            _NETWORKS[key] = ModelSpecs(model, estimator.configs_[0], estimator.inference_config_)
        return _NETWORKS[key]


def _default_weights(kind):
    """Bundled weight file if present (repository checkout with weights/), else download it from Hugging Face."""
    path = DEFAULT_CLASSIFIER_WEIGHTS if kind == "classifier" else DEFAULT_REGRESSOR_WEIGHTS
    if path.is_file():
        return path
    from .hub import HF_REPO_ID, download_weights
    try:
        return download_weights(which=kind)[kind]
    except Exception as error:
        raise FileNotFoundError(
            f"{path.name} is not in {path.parent} and could not be downloaded from Hugging Face ({HF_REPO_ID}). "
            f"Accept the TabPFN-3.5 Non-Commercial License at https://huggingface.co/{HF_REPO_ID}, log in with "
            "`hf auth login` and retry, or pass model_path= to a local copy of the weights.") from error


def _features(X):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_rows, n_features), got shape {X.shape}.")
    return X


def _no_autocast(device):
    return torch.autocast("cuda", enabled=False) if device.startswith("cuda") else nullcontext()


class _WeightedPFN(BaseEstimator):
    _kind = None

    def __init__(self, model_path=None, device="auto", weight_quantile=0.8, random_state=SEED,
                 ignore_pretraining_limits=False):
        self.model_path = model_path
        self.device = device
        self.weight_quantile = weight_quantile
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits

    def _options(self):
        options = dict(n_estimators=1, device=self.device_, random_state=self.random_state, n_preprocessing_jobs=1,
                       fit_mode="fit_preprocessors", inference_precision=torch.float32,
                       ignore_pretraining_limits=self.ignore_pretraining_limits)
        if self._kind == "regressor":
            options["inference_config"] = dict(REGRESSION_INFERENCE_CONFIG)
        return options

    def _context(self, X, y, sample_weight):
        X = _features(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or not len(y):
            raise ValueError("X and y must have the same, non-zero number of rows; y must be 1-D.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != y.shape:
                raise ValueError("sample_weight must have one entry per training row.")
        self.device_ = resolve_device(self.device)
        self.n_features_in_ = X.shape[1]
        self.log_weights_ = None if sample_weight is None else _patch.process_weights(sample_weight, self.weight_quantile)
        _patch.install()
        return X, y

    def _queries(self, X):
        check_is_fitted(self, "log_weights_")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the estimator was fitted with {self.n_features_in_}.")
        return X

    def _run(self, predict_chunk, X):
        with _patch.active_weights(self.log_weights_, prep=True), _no_autocast(self.device_), torch.no_grad():
            return np.concatenate([predict_chunk(X[s:s + QUERY_CHUNK]) for s in range(0, len(X), QUERY_CHUNK)])


class WeightedPFNClassifier(ClassifierMixin, _WeightedPFN):
    """WeightedPFN classifier: TabPFN-3.5 with weighted attention, fine-tuned for 800 steps on synthetic missing-data tasks.

    Parameters
    ----------
    model_path : path to a TabPFN-3.5 safetensors file; None = the fine-tuned classifier (from weights/, or downloaded
        from Hugging Face on first use).
    device : 'auto' (GPU if available, else CPU), 'cpu', 'cuda', 'cuda:1', ...
    weight_quantile : context weights are capped at this quantile before use (0.8 in training); None = no cap.
    random_state : seed of TabPFN's preprocessing (the value used in the experiments of the paper).
    ignore_pretraining_limits : passed to TabPFN; needed on CPU for more than 5,000 training rows.
    """

    _kind = "classifier"

    def fit(self, X, y, sample_weight=None):
        """Store the weighted training context.  ``sample_weight=None`` runs the fine-tuned network without weights."""
        from tabpfn import TabPFNClassifier
        X, y = self._context(X, y, sample_weight)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            self.model_ = None
            return self
        options = self._options()
        spec = _network("classifier", self.model_path or _default_weights("classifier"), options)
        self.model_ = TabPFNClassifier(**dict(options, model_path=spec)).fit(X, y)
        return self

    def predict_proba(self, X):
        X = self._queries(X)
        out = np.zeros((len(X), len(self.classes_)))
        if self.model_ is None:
            out[:] = 1.0
            return out
        out[:, np.searchsorted(self.classes_, self.model_.classes_)] = self._run(self.model_.predict_proba, X)
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class WeightedPFNRegressor(RegressorMixin, _WeightedPFN):
    """WeightedPFN regressor (fine-tuned like the classifier); predicts the weighted conditional mean.

    Parameters are those of :class:`WeightedPFNClassifier`; ``model_path=None`` = the bundled fine-tuned regressor.
    """

    _kind = "regressor"

    def fit(self, X, y, sample_weight=None):
        """Store the weighted training context.  ``sample_weight=None`` runs the fine-tuned network without weights."""
        from tabpfn import TabPFNRegressor
        X, y = self._context(X, y, sample_weight)
        y = y.astype(np.float64, copy=False)
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN/inf: drop rows with a missing outcome and weight the remaining rows.")
        options = self._options()
        spec = _network("regressor", self.model_path or _default_weights("regressor"), options)
        self.model_ = TabPFNRegressor(**dict(options, model_path=spec)).fit(X, y)
        return self

    def predict(self, X):
        X = self._queries(X)
        return self._run(lambda chunk: self.model_.predict(chunk, output_type="mean"), X)
