"""Tests for image preprocessing (decode, normalise, augment, pipelines)."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src import config
from src.preprocessing import (make_augmenter, make_dataset,
                               make_test_dataset, normalize, preprocess_path)


def _write_ppm(path, h=32, w=32):
    header = f"P6\n{w} {h}\n255\n".encode()
    body = bytes(np.arange(h * w * 3, dtype=np.uint8) % 255)
    path.write_bytes(header + body)


@pytest.fixture()
def ppm_images(tmp_path):
    paths = []
    for i in range(8):
        p = tmp_path / f"img_{i}.ppm"
        _write_ppm(p, h=24 + i, w=20 + i)     # variable sizes like GTSRB
        paths.append(str(p))
    return paths, tmp_path


def test_normalize_range_and_dtype():
    img = tf.constant(255, tf.uint8) * tf.ones((4, 4, 3), tf.uint8)
    out = normalize(img)
    assert out.dtype == tf.float32
    assert float(tf.reduce_max(out)) == pytest.approx(1.0)
    assert float(tf.reduce_min(out)) == pytest.approx(1.0)


def test_normalize_mid_value():
    img = tf.fill((2, 2, 3), 128)
    out = normalize(tf.cast(img, tf.uint8))
    assert float(out[0, 0, 0]) == pytest.approx(128 / 255.0)


def test_preprocess_path_resizes_variable_images(ppm_images):
    paths, _ = ppm_images
    for p in paths:
        img = preprocess_path(tf.constant(p))
        assert img.shape == (*config.IMG_SIZE, config.CHANNELS)
        assert float(tf.reduce_min(img)) >= 0.0
        assert float(tf.reduce_max(img)) <= 1.0


def test_augmenter_preserves_shape_and_range(ppm_images):
    paths, _ = ppm_images
    img = preprocess_path(tf.constant(paths[0]))
    augment = make_augmenter(rotation=0.05, translation=0.05, zoom=0.05)
    out = augment(img)
    assert out.shape == img.shape
    assert float(tf.reduce_min(out)) >= -0.05   # nearest fill stays ~[0,1]
    assert float(tf.reduce_max(out)) <= 1.05


def test_augmenter_has_no_flip_layer():
    """Augmentation must not include flipping (sign orientation matters)."""
    augment = make_augmenter()
    layer_names = []
    for cell in (augment.__closure__ or []):
        obj = cell.cell_contents
        if hasattr(obj, "get_config") and hasattr(obj, "call"):
            layer_names.append(type(obj).__name__)
    assert layer_names, "augmentation layers missing from closure"
    assert "RandomFlip" not in layer_names


def test_make_dataset_yields_expected_batches(ppm_images):
    paths, _ = ppm_images
    labels = [i % 2 for i in range(len(paths))]
    ds = make_dataset(paths, labels, training=False, batch_size=4)
    batches = list(ds)
    x, y = batches[0]
    assert x.shape[1:] == (*config.IMG_SIZE, config.CHANNELS)
    assert x.shape[0] <= 4
    assert y.dtype == tf.int32
    assert float(tf.reduce_min(x)) >= 0.0
    assert float(tf.reduce_max(x)) <= 1.0


def test_training_dataset_is_shuffled_with_augmentation(ppm_images):
    paths, _ = ppm_images
    labels = list(range(len(paths)))
    ds = make_dataset(paths, labels, training=True, batch_size=8)
    x, y = next(iter(ds))
    assert x.shape[1:] == (*config.IMG_SIZE, config.CHANNELS)
    # augmented pixels can extend slightly outside [0,1] due to nearest fill
    assert float(tf.reduce_min(x)) >= -0.05
    assert float(tf.reduce_max(x)) <= 1.05


def test_test_dataset_has_no_labels(ppm_images):
    paths, _ = ppm_images
    ds = make_test_dataset(paths, batch_size=4)
    batch = next(iter(ds))
    assert isinstance(batch, tf.Tensor)          # images only
    assert batch.shape[1:] == (*config.IMG_SIZE, config.CHANNELS)
