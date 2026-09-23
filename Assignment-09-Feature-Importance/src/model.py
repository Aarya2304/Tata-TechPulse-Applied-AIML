"""Preprocessing pipeline and Random Forest model.

Leakage safety: every learned transformation (imputation statistics, one-hot
vocabulary) lives inside an sklearn Pipeline fitted on the training split
only. The test split only ever passes through ``transform``.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    """ColumnTransformer + RandomForestRegressor, fitted end-to-end.

    Numeric: median imputation (robust to the skewed numeric distributions).
    Categorical: most-frequent imputation + OneHotEncoder(handle_unknown=
    "ignore") so unseen brands/fuel types at inference time cannot crash the
    pipeline. Tree models do not need scaling, but a StandardScaler on the
    numeric block keeps the pipeline comparable to linear-model baselines.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def regression_metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / R² in original rupee units (zero-safe)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "R2": round(r2, 4)}


def save_metrics(metrics: dict, path=None) -> None:
    path = path or config.MODEL_METRICS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
