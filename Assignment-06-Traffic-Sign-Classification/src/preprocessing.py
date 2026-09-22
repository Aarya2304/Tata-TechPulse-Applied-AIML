"""Image preprocessing: resize/normalisation, tf.data pipelines, augmentation.

Memory design: images stay on disk as variable-size PPM files and are
decoded/resized/normalised lazily inside ``tf.data`` map functions, so no
full-resolution copy of the dataset is ever held in RAM (see README).
"""

from __future__ import annotations

import tensorflow as tf

from src import config
from src.data_loader import decode_image


def normalize(img: tf.Tensor) -> tf.Tensor:
    """uint8 [H,W,3] -> float32 in [0, 1]."""
    return tf.cast(img, tf.float32) / 255.0


def preprocess_path(path: tf.Tensor) -> tf.Tensor:
    """File path -> normalised, resized image tensor."""
    return normalize(decode_image(path))


def make_augmenter(rotation=config.AUGMENTATION["rotation"],
                   translation=config.AUGMENTATION["translation"],
                   zoom=config.AUGMENTATION["zoom"]):
    """Return an augmentation fn. Deliberately NO flips: traffic signs have
    orientation-specific meaning (e.g. turn-left vs turn-right).

    The Keras preprocessing layers are created ONCE here (eagerly), not
    inside the mapped function -- they hold RNG state variables, which
    tf.function forbids creating per call.
    """
    seed = config.RANDOM_STATE
    rot_layer = tf.keras.layers.RandomRotation(
        rotation, fill_mode="nearest", seed=seed)
    trans_layer = tf.keras.layers.RandomTranslation(
        translation, translation, fill_mode="nearest", seed=seed)
    zoom_layer = tf.keras.layers.RandomZoom(
        zoom, fill_mode="nearest", seed=seed)

    def augment(img: tf.Tensor) -> tf.Tensor:
        return zoom_layer(trans_layer(rot_layer(img)))

    return augment


def make_dataset(paths: list[str], labels: list[int] | None = None,
                 training: bool = False,
                 batch_size: int = config.BATCH_SIZE) -> tf.data.Dataset:
    """Build a memory-efficient tf.data pipeline.

    training=True  -> shuffle, light augmentation, prefetch
    training=False -> deterministic order, no augmentation, prefetch
    """
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, l: (preprocess_path(p), l),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: preprocess_path(p),
                    num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        if config.AUGMENT_TRAINING:
            augment = make_augmenter()
            ds = ds.map(lambda img, l: (augment(img), l),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=8192,
                        seed=config.RANDOM_STATE,
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_dataset(paths: list[str],
                      batch_size: int = config.BATCH_SIZE) -> tf.data.Dataset:
    """Paths -> batched normalised images (no labels, no shuffle)."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: preprocess_path(p),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
