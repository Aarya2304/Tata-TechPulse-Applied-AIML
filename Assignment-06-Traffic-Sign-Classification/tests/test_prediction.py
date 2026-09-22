"""Prediction tests with a tiny trained CNN (fast, fully offline)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from src import config
from src.model import build_cnn, compile_model
from src.preprocessing import preprocess_path
from src.predict import load_class_names, predict_image


@pytest.fixture(scope="module")
def tiny_trained_model(tmp_path_factory):
    """One-batch overfit on 8 synthetic 43-class images -> tiny real CNN."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8).astype(
        np.float32) / 255.0
    y = np.arange(8) % config.NUM_CLASSES

    model = compile_model(build_cnn())
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)
    path = tmp_path_factory.mktemp("models") / "tiny.keras"
    model.save(path)
    return path, model


@pytest.fixture()
def ppm_image(tmp_path):
    header = b"P6\n32 32\n255\n"
    body = bytes(np.arange(32 * 32 * 3, dtype=np.uint8) % 255)
    p = tmp_path / "sign.ppm"
    p.write_bytes(header + body)
    return str(p)


def test_predict_output_structure(tiny_trained_model, ppm_image):
    _, model = tiny_trained_model
    class_names = {i: f"class_{i}" for i in range(config.NUM_CLASSES)}
    result = predict_image(model, ppm_image, class_names)
    assert set(result) == {"class_id", "class_name", "confidence",
                           "probabilities"}
    assert 0 <= result["class_id"] < config.NUM_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    probs = result["probabilities"]
    assert probs.shape == (config.NUM_CLASSES,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_preprocessing_matches_training(ppm_image):
    img = preprocess_path(tf.constant(ppm_image))
    assert img.shape == (32, 32, 3)
    assert float(tf.reduce_max(img)) <= 1.0
    assert img.dtype == tf.float32


def test_class_mapping_roundtrip(tmp_path, monkeypatch):
    mapping = {str(i): f"name_{i}" for i in range(config.NUM_CLASSES)}
    path = tmp_path / "class_names.json"
    path.write_text(json.dumps(mapping))
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", path)
    names = load_class_names()
    assert len(names) == config.NUM_CLASSES
    assert all(isinstance(k, int) for k in names)
    assert names[7] == "name_7"


def test_class_mapping_file_exists_after_training():
    """The repository's shipped mapping (if any) must parse correctly."""
    if not config.CLASS_NAMES_PATH.exists():
        pytest.skip("class_names.json not generated yet (train first)")
    names = load_class_names()
    assert len(names) == config.NUM_CLASSES
    assert names[0].startswith("Speed limit (20")


def test_saved_model_loads_and_predicts(tiny_trained_model, ppm_image,
                                        monkeypatch, tmp_path):
    tiny_path, _ = tiny_trained_model
    import tensorflow as tf

    model = tf.keras.models.load_model(tiny_path)
    class_path = tmp_path / "class_names.json"
    class_path.write_text(json.dumps(
        {str(i): f"c{i}" for i in range(config.NUM_CLASSES)}))
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", class_path)

    result = predict_image(model, ppm_image, load_class_names())
    assert isinstance(result["class_id"], int)
    assert result["confidence"] >= float(
        np.max(result["probabilities"]) - 1e-6)


def test_predict_main_with_image(tiny_trained_model, ppm_image, monkeypatch,
                                 capsys, tmp_path):
    """CLI happy path: python -m src.predict image.ppm"""
    from src import predict as predict_module
    tiny_path, _ = tiny_trained_model
    class_path = tmp_path / "class_names.json"
    class_path.write_text(json.dumps(
        {str(i): f"c{i}" for i in range(config.NUM_CLASSES)}))
    monkeypatch.setattr(config, "MODEL_PATH", tiny_path)
    monkeypatch.setattr(config, "CLASS_NAMES_PATH", class_path)
    monkeypatch.setattr("sys.argv", ["predict.py", ppm_image])
    predict_module.main()
    out = capsys.readouterr().out
    assert "Predicted class ID" in out
    assert "Predicted class name" in out
    assert "Confidence" in out


def test_predict_main_missing_model(monkeypatch, tmp_path):
    """Clear error when the model has not been trained yet."""
    from src import predict as predict_module
    monkeypatch.setattr(config, "MODEL_PATH",
                        tmp_path / "nonexistent.keras")
    monkeypatch.setattr("sys.argv", ["predict.py"])
    with pytest.raises(SystemExit):
        predict_module.main()
