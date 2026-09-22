"""Metrics helpers: MAE, RMSE, R^2 and grouped cross-validation reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from src import config


def regression_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R^2 as plain floats."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_model(model, X, y) -> dict:
    """5-fold shuffled CV (same seed everywhere); returns per-metric mean/std."""
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True,
                  random_state=config.RANDOM_STATE)
    scores = cross_validate(
        model, X, y,
        cv=kfold,
        scoring={"MAE": "neg_mean_absolute_error",
                 "RMSE": "neg_root_mean_squared_error",
                 "R2": "r2"},
        n_jobs=-1,
    )
    return {
        metric: {
            "mean": float(-scores[f"test_{metric}"].mean())
            if metric != "R2" else float(scores["test_R2"].mean()),
            "std": float(scores[f"test_{metric}"].std()),
        }
        for metric in ("MAE", "RMSE", "R2")
    }


def metrics_to_frame(rows: list) -> pd.DataFrame:
    """Build a tidy one-row-per-model comparison table."""
    return pd.DataFrame(rows).set_index("model")
