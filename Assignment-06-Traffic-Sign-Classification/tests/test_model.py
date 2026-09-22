"""Tests for the CNN architecture, forward pass and persistence."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src import config
from src.model import build_cnn, compile_model, make_callbacks


@pytest.fixture(scope="module")
def model():
    return compile_model(build_cnn())


def test_input_output_shapes(model):
    assert model.input_shape == (None, *config.IMG_SIZE, config.CHANNELS)
    assert model.output_shape == (None, config.NUM_CLASSES)


def test_forward_pass_probabilities_sum_to_one(model):
    x = tf.random.uniform((4, *config.IMG_SIZE, config.CHANNELS), seed=42)
    y = model(x, training=False)
    assert y.shape == (4, config.NUM_CLASSES)
    assert np.allclose(y.numpy().sum(axis=1), 1.0, atol=1e-5)
    assert float(tf.reduce_min(y)) >= 0.0        # softmax outputs


def test_architecture_has_expected_blocks(model):
    names = [type(l).__name__ for l in model.layers]
    conv = names.count("Conv2D")
    pool = names.count("MaxPooling2D")
    bn = names.count("BatchNormalization")
    assert conv == 6 and pool == 3 and bn == 6   # 2 convs per block x 3
    assert "GlobalAveragePooling2D" in names
    assert "Dropout" in names
    # filter progression contains 32/64/128 channels somewhere
    filters = [l.filters for l in model.layers
               if isinstance(l, tf.keras.layers.Conv2D)]
    assert set(filters) == {32, 64, 128}


def test_final_activation_is_softmax(model):
    out_layer = model.layers[-1]
    assert out_layer.activation.__name__ == "softmax"


def test_compile_settings(model):
    assert model.loss == "sparse_categorical_crossentropy"
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert model.compiled_metrics is not None


def test_model_has_no_horizontal_flip():
    """Architecture must not embed flip augmentation (orientation matters)."""
    for layer in build_cnn().layers:
        flip_types = ("RandomFlip",)
        assert type(layer).__name__ not in flip_types


def test_callbacks_config(tmp_path):
    callbacks = make_callbacks(tmp_path / "m.keras")
    kinds = [type(c).__name__ for c in callbacks]
    assert "ModelCheckpoint" in kinds
    assert "EarlyStopping" in kinds
    assert "ReduceLROnPlateau" in kinds
    ckpt = next(c for c in callbacks
                if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
    assert ckpt.save_best_only and ckpt.monitor == "val_accuracy"
    es = next(c for c in callbacks
              if isinstance(c, tf.keras.callbacks.EarlyStopping))
    assert es.restore_best_weights


def test_model_save_load_roundtrip(tmp_path, model):
    path = tmp_path / "roundtrip.keras"
    x = tf.random.uniform((2, *config.IMG_SIZE, config.CHANNELS), seed=1)
    before = model(x, training=False).numpy()
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    after = loaded(x, training=False).numpy()
    assert np.allclose(before, after, atol=1e-6)


def test_model_parameter_count_is_reasonable(model):
    """Compact CNN: a few million parameters at most (CPU-friendly)."""
    total = model.count_params()
    assert 50_000 < total < 5_000_000
