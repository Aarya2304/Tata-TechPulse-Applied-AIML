"""Model construction and evaluation for the vehicle-price regression task.

Keeping construction here (separate from training and serving) makes the
MLOps lifecycle explicit: this module defines *what* is trained,
``train.py`` performs *the training + tracking*, and ``inference.py`` /
``app.py`` handle *serving*.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def build_model(
    n_estimators: int = config.N_ESTIMATORS,
    random_state: int = config.RANDOM_STATE,
) -> RandomForestRegressor:
    """Create the RandomForestRegressor used across the whole project.

    Deterministic by construction: a fixed ``random_state`` and a fixed
    feature order (``config.FEATURE_NAMES``) make repeated training runs
    on the same data produce identical predictions.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Return MAE / RMSE / R2 and fail loudly on non-finite values.

    RMSE is computed as ``sqrt(MSE)`` directly so the implementation is
    stable across scikit-learn versions.
    """
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError(f"non-finite evaluation metrics: {metrics}")
    return metrics
