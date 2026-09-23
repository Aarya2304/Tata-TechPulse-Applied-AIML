"""Tests for the inference layer (loading, prediction, metadata)."""

from __future__ import annotations

import math

import pytest

from src import config, inference

SAMPLE_FEATURES = {
    "vehicle_age": 5,
    "km_driven": 45_000,
    "mileage_kmpl": 18.5,
    "engine_cc": 1_498,
    "max_power_bhp": 100,
}


@pytest.fixture(scope="module")
def trained_model():
    """Load (or deterministically bootstrap) the deployment model once."""
    inference.reset_model_cache()
    return inference.ensure_model()


def test_inference_model_loads_successfully(trained_model):
    assert trained_model is not None
    assert hasattr(trained_model, "predict")


def test_prediction_returns_numeric_value(trained_model):
    value = inference.predict_price(SAMPLE_FEATURES)
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_prediction_probability_style_value_is_positive(trained_model):
    assert inference.predict_price(SAMPLE_FEATURES) > 0


def test_feature_order_is_respected(trained_model):
    """Shuffled key order must not silently produce the default order."""
    shuffled = dict(reversed(list(SAMPLE_FEATURES.items())))
    assert inference.predict_price(shuffled) == pytest.approx(
        inference.predict_price(SAMPLE_FEATURES)
    )


def test_newer_vehicle_prices_higher(trained_model):
    newer = {**SAMPLE_FEATURES, "vehicle_age": 1}
    older = {**SAMPLE_FEATURES, "vehicle_age": 15}
    assert inference.predict_price(newer) >= inference.predict_price(older)


def test_metadata_contains_required_fields(trained_model):
    metadata = inference.get_metadata()
    for key in (
        "model_type",
        "random_state",
        "training_timestamp",
        "dataset_type",
        "feature_names",
        "metrics",
        "application_version",
    ):
        assert key in metadata, f"metadata missing '{key}'"
    assert metadata["feature_names"] == config.FEATURE_NAMES


def test_metadata_version_matches_central_version(trained_model):
    assert inference.get_metadata()["application_version"] == config.APP_VERSION


def test_metadata_records_mlflow_run(trained_model):
    assert inference.get_metadata().get("mlflow_run_id")


def test_reset_model_cache_reloads(trained_model):
    inference.reset_model_cache()
    reloaded = inference.ensure_model()
    assert reloaded is not None
    assert inference.predict_price(SAMPLE_FEATURES) == pytest.approx(
        inference.predict_price(SAMPLE_FEATURES)
    )
