"""Model zoo: four sklearn regression pipelines on the log1p target.

Each model is a full ``Pipeline(preprocessor, regressor)`` wrapped in a
``TransformedTargetRegressor(func=log1p, inverse_func=expm1)`` so that:

* the model fits on log1p(selling_price) -- appropriate for the strongly
  right-skewed price distribution;
* ``model.predict(X)`` already returns prices in ORIGINAL rupee units
  (inverse-transform is built into the wrapper), so all reported metrics
  are in price space, never log space.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_preprocessor


def build_models() -> dict:
    """Return name -> fitted-ready estimator for all candidates."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            random_state=config.RANDOM_STATE),
    }
    wrapped = {}
    for name, reg in models.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", reg),
        ])
        wrapped[name] = TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return wrapped


def get_pipeline(model) -> Pipeline:
    """Return the inner sklearn Pipeline of a (possibly unfitted) wrapper."""
    if hasattr(model, "regressor_"):
        return model.regressor_
    if hasattr(model, "regressor"):
        return model.regressor
    return model
