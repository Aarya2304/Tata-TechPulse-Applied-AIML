"""End-to-end pipeline tests on a small subset (fast, deterministic)."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_raw
from src.evaluation import classification_metrics, select_best
from src.feature_engineering import build_feature_frame
from src.models import build_models, build_pipeline


@pytest.fixture(scope="module")
def split_data():
    df = load_raw()
    X, y = build_feature_frame(df)
    X = X.sample(n=600, random_state=config.RANDOM_STATE)
    y = y.loc[X.index]
    return train_test_split(X, y, test_size=0.25,
                            random_state=config.RANDOM_STATE, stratify=y)


@pytest.fixture(scope="module")
def trained(split_data):
    X_train, X_test, y_train, y_test = split_data
    models = build_models()
    fitted = {}
    for name in ("Logistic Regression", "Random Forest"):
        pipe = build_pipeline(models[name])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted, (X_train, X_test, y_train, y_test)


def test_all_three_models_defined():
    models = build_models()
    assert set(models) == {"Logistic Regression", "Random Forest",
                           "Hist Gradient Boosting"}


def test_class_weighting_enabled():
    """All three models must use class_weight='balanced' (no resampling)."""
    for name, model in build_models().items():
        assert model.class_weight == "balanced", name


def test_pipeline_trains_on_subset(trained):
    fitted, _ = trained
    for name, pipe in fitted.items():
        assert hasattr(pipe, "predict_proba")


def test_prediction_shape_and_type(trained, split_data):
    fitted, (_, X_test, _, _) = trained
    for pipe in fitted.values():
        prob = pipe.predict_proba(X_test)
        pred = pipe.predict(X_test)
        assert prob.shape == (len(X_test), 2)
        assert pred.shape == (len(X_test),)
        assert set(pred) <= {0, 1}
        assert np.isfinite(prob).all()
        assert np.allclose(prob.sum(axis=1), 1.0)


def test_metrics_are_finite_and_sane(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipe = build_pipeline(build_models()["Random Forest"])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    m = classification_metrics(y_test, prob)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["average_precision"] <= 1.0
    assert m["f1"] >= 0.0
    assert m["true_positives"] + m["false_negatives"] == int(
        (y_test == 1).sum())


def test_select_best_uses_predefined_criterion():
    comparison = pd.DataFrame(
        {"f1": [0.5, 0.8, 0.8],
         "average_precision": [0.9, 0.7, 0.9],
         "recall": [0.4, 0.5, 0.5]},
        index=["A", "B", "C"])
    assert select_best(comparison) == "C"   # F1 tie -> AP tie-break


def test_model_saves_and_reloads(trained, split_data, tmp_path):
    fitted, (_, X_test, _, _) = trained
    path = tmp_path / "model.joblib"
    joblib.dump(fitted["Random Forest"], path)
    reloaded = joblib.load(path)
    original = fitted["Random Forest"].predict_proba(X_test)[:, 1]
    assert np.allclose(reloaded.predict_proba(X_test)[:, 1], original)


def test_target_never_in_features():
    assert config.TARGET not in config.FEATURES
    for col in config.LEAKAGE_COLUMNS + config.ID_COLUMNS:
        assert col not in config.FEATURES


def test_predict_cli_demo_mode(trained, tmp_path, monkeypatch, capsys):
    from src import predict as predict_module
    fitted, _ = trained
    model_path = tmp_path / "model.joblib"
    joblib.dump(fitted["Random Forest"], model_path)
    monkeypatch.setattr(config, "BEST_MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "config", config)
    monkeypatch.setattr("sys.argv", ["predict.py"])
    predict_module.main()
    out = capsys.readouterr().out
    assert "failure probability" in out
    assert "NO FAILURE" in out or "FAILURE" in out


def test_predict_cli_csv_mode(trained, tmp_path, monkeypatch, capsys):
    from src import predict as predict_module
    fitted, _ = trained
    model_path = tmp_path / "model.joblib"
    joblib.dump(fitted["Random Forest"], model_path)

    csv_in = tmp_path / "sensors.csv"
    pd.DataFrame(predict_module.DEMO_MACHINES).to_csv(csv_in, index=False)
    monkeypatch.setattr(config, "BEST_MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "config", config)
    monkeypatch.setattr("sys.argv", ["predict.py", "--csv", str(csv_in)])
    predict_module.main()
    out = capsys.readouterr().out
    assert "predictions" in out
    preds = pd.read_csv(tmp_path / "sensors_predictions.csv")
    assert len(preds) == len(predict_module.DEMO_MACHINES)
    assert preds["failure_probability"].between(0, 1).all()
    assert set(preds["predicted_failure"]) <= {0, 1}
