"""Download the weight files from a Hugging Face model repository (for installs without the bundled weights/ folder)."""
from __future__ import annotations

from pathlib import Path

from .estimators import DEFAULT_CLASSIFIER_WEIGHTS, DEFAULT_REGRESSOR_WEIGHTS

HF_REPO_ID = "xiajunwen/WeightedPFN"  # gated Hugging Face model repository with the two weight files
WEIGHT_FILES = {"classifier": DEFAULT_CLASSIFIER_WEIGHTS.name, "regressor": DEFAULT_REGRESSOR_WEIGHTS.name}


def download_weights(repo_id=HF_REPO_ID, *, which=("classifier", "regressor"), revision=None, cache_dir=None, token=None):
    """Download the weights from Hugging Face (``xiajunwen/WeightedPFN`` by default; cached locally) and return ``{"classifier": path, "regressor": path}``.

    The repository is gated: accept the license on its web page first and log in (``hf auth login``) or pass ``token``.
    The estimators call this automatically when the bundled ``weights/`` folder is absent; the returned paths can also be
    passed to ``model_path=`` of :class:`WeightedPFNClassifier` / :class:`WeightedPFNRegressor`.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError("download_weights needs huggingface_hub: pip install huggingface_hub") from error
    which = (which,) if isinstance(which, str) else tuple(which)
    unknown = set(which) - WEIGHT_FILES.keys()
    if unknown:
        raise ValueError(f"which must be among {sorted(WEIGHT_FILES)}, got {sorted(unknown)}")
    return {kind: Path(hf_hub_download(repo_id, WEIGHT_FILES[kind], revision=revision, cache_dir=cache_dir, token=token))
            for kind in which}
