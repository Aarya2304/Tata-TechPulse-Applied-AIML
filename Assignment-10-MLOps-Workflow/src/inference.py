"""Inference layer: load the trained model and predict vehicle prices.

Loads the small joblib deployment artifact exported by ``src.train``.
When the artifact is missing (fresh clone / CI), :func:`ensure_model`
bootstraps it deterministically from the tiny smoke configuration, so
the API and the container never depend on a model sitting only on the
developer's machine.  The bootstrap is explicitly flagged in
``model_metadata.json``.
"""

from __future__ import annotations

import json
import threading

from src import config

_model = None
_metadata: dict | None = None
_lock = threading.Lock()


def _bootstrap_artifact() -> None:
    """Create the deployment artifact deterministically (smoke size)."""
    # Imported lazily: keeps inference import light and avoids a
    # train->inference import cycle at module load time.
    from src.train import train_and_log

    train_and_log(
        n_samples=config.SMOKE_N_SAMPLES,
        n_estimators=config.SMOKE_N_ESTIMATORS,
        seed=config.RANDOM_STATE,
    )


def ensure_model():
    """Return the loaded model, training a smoke model if none exists."""
    global _model, _metadata
    with _lock:
        if _model is not None:
            return _model
        if not config.MODEL_PATH.exists():
            _bootstrap_artifact()
        import joblib

        _model = joblib.load(config.MODEL_PATH)
        if config.METADATA_PATH.exists():
            _metadata = json.loads(
                config.METADATA_PATH.read_text(encoding="utf-8")
            )
        return _model


def get_metadata() -> dict:
    """Model metadata (model_type, metrics, run id, version)."""
    ensure_model()
    return _metadata or {}


def predict_price(features: dict) -> float:
    """Predict selling price for one vehicle feature dict."""
    model = ensure_model()
    row = [[float(features[name]) for name in config.FEATURE_NAMES]]
    prediction = model.predict(row)
    return float(prediction[0])


def reset_model_cache() -> None:
    """Forget the cached model (used by tests and CI)."""
    global _model, _metadata
    with _lock:
        _model = None
        _metadata = None
