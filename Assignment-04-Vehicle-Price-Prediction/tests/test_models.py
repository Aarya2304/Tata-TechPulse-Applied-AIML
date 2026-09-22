"""Tests for the model zoo: fitting, shapes, finiteness, leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.models import build_models, get_pipeline

EXPECTED_MODELS = ["Linear Regression", "Ridge Regression", "Random Forest",
                   "Gradient Boosting"]


@pytest.fixture(scope="module")
def tiny():
    """Small but multi-category training set -> fast fits."""
    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({
        "vehicle_age": rng.integers(0, 12, n).astype(float),
        "km_driven": rng.integers(5_000, 120_000, n).astype(float),
        "mileage_kmpl": rng.normal(19, 3, n),
        "engine_cc": rng.choice([998.0, 1248.0, 1498.0], n),
        "max_power_bhp": rng.normal(85, 20, n),
        "torque_nm": rng.normal(160, 40, n),
        "seats": rng.choice([5.0, 7.0], n),
        "km_per_year": rng.normal(12_000, 3_000, n),
        "fuel": rng.choice(["Petrol", "Diesel"], n),
        "seller_type": rng.choice(["Individual", "Dealer"], n),
        "transmission": rng.choice(["Manual", "Automatic"], n),
        "owner": rng.choice(["First Owner", "Second Owner"], n),
        "brand": rng.choice(["Maruti", "Hyundai", "Toyota"], n),
    })
    base = (2e5 - 8e3 * X["vehicle_age"] + 0.6 * X["max_power_bhp"] * 1e3
            + np.where(X["fuel"] == "Diesel", 4e4, 0))
    y = pd.Series(base + rng.normal(0, 1e4, n), name="selling_price")
    return X, y


def test_all_four_models_present():
    models = build_models()
    assert sorted(models.keys()) == sorted(EXPECTED_MODELS)


def test_models_are_wrapped_pipelines():
    for name, model in build_models().items():
        assert isinstance(model, TransformedTargetRegressor)
        assert model.func is np.log1p and model.inverse_func is np.expm1
        inner = get_pipeline(model)      # works on unfitted wrappers too
        assert isinstance(inner, Pipeline)
        assert "preprocessor" in inner.named_steps


def test_each_model_fits_and_predicts(tiny):
    X, y = tiny
    for name, model in build_models().items():
        model.fit(X, y)
        pred = model.predict(X.head(7))
        assert pred.shape == (7,)
        assert np.isfinite(pred).all()
        assert (pred > 0).all()   # expm1 of finite log-prices stays positive


def test_predictions_are_in_rupee_space(tiny):
    """The TTR wrapper must return prices, not log-prices."""
    X, y = tiny
    model = build_models()["Linear Regression"].fit(X, y)
    pred = model.predict(X)
    assert pred.min() > 1000        # log-space values would be ~12-14
    assert pred.max() < 1e8


def test_target_never_in_model_inputs(tiny):
    X, _ = tiny
    for name, model in build_models().items():
        pre = get_pipeline(model).named_steps["preprocessor"]
        used = set()
        for _, _, cols in pre.transformers:
            if isinstance(cols, list):
                used.update(cols)
        assert "selling_price" not in used
        assert not (set(X.columns) & {"selling_price"})


def test_preprocessor_fitted_on_train_rows_only(tiny):
    """Statistics inside the pipeline come from the fit call's rows."""
    X, y = tiny
    train_X, test_X = X.iloc[:40], X.iloc[40:]
    train_y, _ = y.iloc[:40], y.iloc[40:]

    model = build_models()["Ridge Regression"]
    model.fit(train_X, train_y)
    pre = get_pipeline(model).named_steps["preprocessor"]
    scaler = pre.named_transformers_["num"].named_steps["scaler"]

    train_mean_age = train_X["vehicle_age"].mean()
    assert scaler.mean_[0] == pytest.approx(train_mean_age)

    # touching test rows afterwards cannot change learned stats
    _ = model.predict(test_X)
    assert scaler.mean_[0] == pytest.approx(train_mean_age)
