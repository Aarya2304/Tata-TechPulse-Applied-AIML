"""Tests for the preprocessing pipeline and Random Forest model."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.data import load_and_prepare, make_splits
from src.model import build_pipeline, regression_metrics


@pytest.fixture(scope="module")
def fitted():
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = make_splits(X, y)
    pipeline = build_pipeline().fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def test_pipeline_structure(fitted):
    pipeline, *_ = fitted
    pre = pipeline.named_steps["preprocessor"]
    assert "num" in pre.named_transformers_ and "cat" in pre.named_transformers_
    # OneHotEncoder must ignore unseen categories at transform time
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    assert isinstance(onehot, OneHotEncoder) and onehot.handle_unknown == "ignore"


def test_preprocessing_is_fitted_on_train_only(fitted):
    """Imputer statistics must come from the training split, not the full data."""
    pipeline, X_train, X_test, y_train, y_test = fitted
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    # compare against the training median of one numeric column with NaNs
    col = "seats" if X_train["seats"].isna().any() else "mileage_kmpl"
    expected = X_train[col].median()
    col_idx = _numeric_index(pipeline, col)
    assert np.isclose(imputer.statistics_[col_idx], expected)


NUMERIC_ORDER = [
    "year", "km_driven", "seats", "mileage_kmpl", "engine_cc",
    "max_power_bhp", "torque_nm", "vehicle_age", "km_per_year",
]


def _numeric_index(pipeline, column):
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    return names.index(column)


def test_model_trains_and_predicts(fitted):
    pipeline, _, X_test, _, _ = fitted
    preds = pipeline.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert np.isfinite(preds).all()
    assert (preds > 0).all()  # prices are positive


def test_metrics_are_finite_and_sensible(fitted):
    pipeline, _, X_te, _, y_te = fitted
    metrics = regression_metrics(y_te, pipeline.predict(X_te))
    for key in ("MAE", "RMSE", "R2"):
        assert np.isfinite(metrics[key])
    assert 0.5 < metrics["R2"] <= 1.0
    assert 0 < metrics["MAE"] < metrics["RMSE"]


def test_predictions_deterministic_with_seed():
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = make_splits(X, y)
    p1 = build_pipeline().fit(X_train, y_train).predict(X_test.head(20))
    p2 = build_pipeline().fit(X_train, y_train).predict(X_test.head(20))
    np.testing.assert_allclose(p1, p2)


def test_unknown_category_does_not_crash(fitted):
    """A brand/owner value never seen in training must transform cleanly."""
    pipeline, X_train, _, _, _ = fitted
    weird = X_train.head(2).copy()
    weird["brand"] = "Zephyr Motors"
    weird["owner"] = "Ninth Owner"
    preds = pipeline.predict(weird)
    assert np.isfinite(preds).all()
