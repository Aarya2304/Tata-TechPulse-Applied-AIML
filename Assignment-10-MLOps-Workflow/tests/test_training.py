"""Tests for dataset generation and the deterministic training workflow.

All tests use the smoke configuration so the suite stays fast; none of
them require Docker, the full dataset or a live MLflow server.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config
from src.data import generate_dataset, load_training_data, validate_dataset
from src.model import build_model, evaluate_regression
from src.train import train_and_log, train_model


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset()


@pytest.fixture(scope="module")
def trained():
    """Train the smoke model once and reuse it across tests."""
    return train_model(
        n_samples=config.SMOKE_N_SAMPLES,
        n_estimators=config.SMOKE_N_ESTIMATORS,
        seed=config.RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_generation_is_deterministic():
    first = generate_dataset()
    second = generate_dataset()
    pd.testing.assert_frame_equal(first, second)


def test_dataset_generation_varies_with_seed():
    assert not generate_dataset(seed=1).equals(generate_dataset(seed=2))


def test_dataset_has_expected_columns(dataset):
    expected = config.FEATURE_NAMES + [config.TARGET_NAME]
    assert list(dataset.columns) == expected


def test_dataset_has_no_missing_values(dataset):
    validate_dataset(dataset)  # raises on any structural problem
    assert int(dataset.isna().sum().sum()) == 0
    assert np.isfinite(dataset.to_numpy(dtype=float)).all()


def test_dataset_size_matches_configuration(dataset):
    assert len(dataset) == config.N_SAMPLES


def test_dataset_values_are_plausible(dataset):
    assert (dataset["vehicle_age"] >= 0).all()
    assert (dataset["km_driven"] > 0).all()
    assert (dataset["mileage_kmpl"] > 0).all()
    assert (dataset["engine_cc"] > 0).all()
    assert (dataset["max_power_bhp"] > 0).all()
    assert (dataset[config.TARGET_NAME] > 0).all()


def test_dataset_encodes_market_intuition(dataset):
    """Newer vehicles should not be cheaper on average in the generator."""
    young = dataset.loc[dataset["vehicle_age"] <= 3, config.TARGET_NAME].median()
    old = dataset.loc[dataset["vehicle_age"] >= 12, config.TARGET_NAME].median()
    assert young >= old


# ---------------------------------------------------------------------------
# Split + model
# ---------------------------------------------------------------------------


def test_train_test_split_is_reproducible():
    X, y = load_training_data(n_samples=config.SMOKE_N_SAMPLES)
    s1 = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    s2 = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    for a, b in zip(s1, s2):
        pd.testing.assert_index_equal(a.index, b.index)


def test_model_can_train_and_predict(trained):
    reg, metrics, (_, X_test, _, _) = trained
    predictions = reg.predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
    assert all(np.isfinite(v) for v in metrics.values())


def test_evaluation_metrics_are_finite_and_ordered(trained):
    _, metrics, (_, _, _, y_test) = trained
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["rmse"] >= metrics["mae"]  # true for every distribution
    assert -1.0 <= metrics["r2"] <= 1.0
    assert len(y_test) == int(config.SMOKE_N_SAMPLES * config.TEST_SIZE)


def test_training_is_reproducible_across_runs(trained):
    """Same seed -> identical metrics and identical predictions."""
    reg, metrics, (_, X_test, _, _) = trained
    reg2, metrics2, _ = train_model(
        n_samples=config.SMOKE_N_SAMPLES,
        n_estimators=config.SMOKE_N_ESTIMATORS,
        seed=config.RANDOM_STATE,
    )
    assert metrics == metrics2
    np.testing.assert_array_equal(reg.predict(X_test), reg2.predict(X_test))


def test_build_model_uses_requested_settings():
    reg = build_model(n_estimators=7, random_state=99)
    assert reg.n_estimators == 7
    assert reg.random_state == 99


# ---------------------------------------------------------------------------
# MLflow workflow
# ---------------------------------------------------------------------------


def test_mlflow_training_workflow_executes(tmp_path):
    """Full train+log cycle against an isolated temporary MLflow store."""
    result = train_and_log(
        n_samples=config.SMOKE_N_SAMPLES,
        n_estimators=config.SMOKE_N_ESTIMATORS,
        seed=config.RANDOM_STATE,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    assert result["run_id"]
    metrics = result["metrics"]
    assert all(np.isfinite(v) for v in metrics.values())

    # The run must actually contain params and metrics.
    from mlflow.tracking import MlflowClient

    run = MlflowClient(
        tracking_uri=(tmp_path / "mlruns").as_uri()
    ).get_run(result["run_id"])
    assert run.data.params["model_type"] == config.MODEL_TYPE
    assert run.data.params["n_estimators"] == str(config.SMOKE_N_ESTIMATORS)
    assert run.data.params["random_state"] == str(config.RANDOM_STATE)
    assert float(run.data.metrics["r2"]) == pytest.approx(metrics["r2"])


def test_training_writes_deployment_artifacts(tmp_path):
    """The real artifact paths exist after a run (smoke-sized)."""
    train_and_log(
        n_samples=config.SMOKE_N_SAMPLES,
        n_estimators=config.SMOKE_N_ESTIMATORS,
        seed=config.RANDOM_STATE,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
    )
    assert config.MODEL_PATH.exists()
    assert config.METADATA_PATH.exists()
    assert config.METRICS_PATH.exists()
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["feature_names"] == config.FEATURE_NAMES
    assert metadata["application_version"] == config.APP_VERSION
