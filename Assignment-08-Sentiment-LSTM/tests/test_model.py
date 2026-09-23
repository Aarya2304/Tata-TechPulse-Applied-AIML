"""Tests for the LSTM model (tiny vocabulary, CPU-fast)."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.model import build_lstm_model


@pytest.fixture(scope="module")
def small_model():
    return build_lstm_model(vocab_size=50, sequence_length=16,
                            embedding_dim=8, lstm_units=8, dense_units=4)


def test_build_lstm_model_returns_compiled_sequential(small_model):
    from tensorflow.keras import Sequential

    assert isinstance(small_model, Sequential)
    assert small_model.optimizer is not None          # compiled
    assert small_model.loss == "binary_crossentropy"


def test_lstm_layer_present_and_no_transformers(small_model):
    """The model must contain an LSTM layer (and nothing transformer-like)."""
    names = [type(layer).__name__ for layer in small_model.layers]
    assert "LSTM" in names
    assert "Embedding" in names
    assert not any("Attention" in n or "Transformer" in n for n in names)


def test_forward_pass_output_shape(small_model):
    X = np.random.randint(0, 50, size=(4, 16))
    y = small_model.predict(X, verbose=0)
    assert y.shape == (4, 1)


def test_output_probabilities_in_unit_range(small_model):
    X = np.random.randint(0, 50, size=(6, 16))
    y = small_model.predict(X, verbose=0).reshape(-1)
    assert ((y >= 0.0) & (y <= 1.0)).all()


def test_model_can_fit_tiny_dataset():
    """One gradient step must run and reduce loss deterministically."""
    tf.keras.utils.set_random_seed(42)
    model = build_lstm_model(vocab_size=30, sequence_length=8,
                             embedding_dim=8, lstm_units=8, dense_units=4)
    X = np.random.randint(0, 30, size=(8, 8))
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
    loss0 = model.evaluate(X, y, verbose=0)
    model.train_on_batch(X, y)
    loss1 = model.evaluate(X, y, verbose=0)
    assert loss1 < loss0  # training moves in the right direction


def test_model_summary_mentions_expected_layers():
    model = build_lstm_model()
    lines = []
    model.summary(print_fn=lines.append)
    text = "\n".join(lines)
    for expected in ("embedding", "lstm", "dense"):
        assert expected in text.lower()
