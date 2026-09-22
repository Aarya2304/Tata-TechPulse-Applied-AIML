"""End-to-end pipeline tests on a fast subset (small data, few trees)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_and_prepare
from src.evaluation import regression_metrics
from src.feature_engineering import add_km_per_year, add_vehicle_age
from src.models import build_models


@pytest.fixture(scope="module")
def split_data():
    df = load_and_prepare(verbose=False)
    df = add_vehicle_age(df)
    df = add_km_per_year(df)
    df = df.sample(n=400, random_state=config.RANDOM_STATE)
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=0.25,
                            random_state=config.RANDOM_STATE)


@pytest.fixture(scope="module")
def trained_models(split_data):
    X_train, X_test, y_train, y_test = split_data
    models = build_models()
    for name in ("Linear Regression", "Random Forest"):
        models[name].fit(X_train, y_train)
    return models, (X_train, X_test, y_train, y_test)


def test_end_to_end_metrics_are_finite(trained_models):
    models, (_, X_test, _, y_test) = trained_models
    for name in ("Linear Regression", "Random Forest"):
        pred = models[name].predict(X_test)
        m = regression_metrics(y_test, pred)
        for value in m.values():
            assert np.isfinite(value)
        assert 0.0 <= m["R2"] <= 1.0


def test_random_forest_beats_random_guessing(trained_models):
    """Sanity: the model must beat predicting the training mean."""
    models, (X_train, X_test, y_train, y_test) = trained_models
    pred = models["Random Forest"].predict(X_test)
    naive_mae = np.mean(np.abs(y_test - y_train.mean()))
    model_mae = np.mean(np.abs(y_test - pred))
    assert model_mae < naive_mae


def test_model_saves_and_reloads(trained_models, tmp_path):
    models, (_, X_test, _, _) = trained_models
    path = tmp_path / "model.joblib"
    joblib.dump(models["Random Forest"], path)
    reloaded = joblib.load(path)
    original = models["Random Forest"].predict(X_test)
    assert np.allclose(reloaded.predict(X_test), original)


def test_saved_artifacts_consistency(trained_models, tmp_path):
    """metrics.json-style payload must carry rupee-space finite values."""
    models, (_, X_test, _, y_test) = trained_models
    pred = models["Random Forest"].predict(X_test)
    m = regression_metrics(y_test, pred)
    payload = {"selected_model": "Random Forest",
               "target_transform": config.TARGET_TRANSFORM,
               **{f"test_{k}": v for k, v in m.items()}}
    with open(tmp_path / "metrics.json", "w") as fh:
        json.dump(payload, fh)
    with open(tmp_path / "metrics.json") as fh:
        loaded = json.load(fh)
    assert loaded["selected_model"] == "Random Forest"
    assert loaded["target_transform"] == "log1p"
    assert np.isfinite(loaded["test_R2"])


def test_predict_cli_demo_mode(trained_models, tmp_path, monkeypatch,
                               capsys):
    """The prediction CLI works end-to-end against a saved model."""
    from src import predict as predict_module

    models, _ = trained_models
    model_path = tmp_path / "model.joblib"
    joblib.dump(models["Random Forest"], model_path)

    monkeypatch.setattr(config, "BEST_MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "config", config)
    monkeypatch.setattr(
        "sys.argv", ["predict.py"])

    predict_module.main()
    out = capsys.readouterr().out
    assert "Predicted vehicle price" in out
    assert "Rs." in out


def test_predict_cli_csv_mode(trained_models, tmp_path, monkeypatch, capsys):
    from src import predict as predict_module

    models, _ = trained_models
    model_path = tmp_path / "model.joblib"
    joblib.dump(models["Random Forest"], model_path)

    csv_in = tmp_path / "cars.csv"
    pd.DataFrame(predict_module.DEMO_VEHICLES).to_csv(csv_in, index=False)

    monkeypatch.setattr(config, "BEST_MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "config", config)
    monkeypatch.setattr(
        "sys.argv", ["predict.py", "--csv", str(csv_in)])

    predict_module.main()
    out = capsys.readouterr().out
    assert "predictions" in out
    out_csv = tmp_path / "cars_predictions.csv"
    assert out_csv.exists()
    preds = pd.read_csv(out_csv)
    assert len(preds) == len(predict_module.DEMO_VEHICLES)
    assert preds["predicted_price_inr"].gt(0).all()
