"""Compact CNN for GTSRB (CPU-friendly), built with TensorFlow/Keras."""

from __future__ import annotations

import tensorflow as tf

from src import config


def build_cnn(input_shape: tuple[int, int, int] = (*config.IMG_SIZE,
                                                    config.CHANNELS),
              num_classes: int = config.NUM_CLASSES) -> tf.keras.Model:
    """Compact 3-block CNN: 32 -> 64 -> 128 filters, then a dense head.

    Augmentation lives in the tf.data pipeline (src/preprocessing.py), not
    inside the model, so saved-model inference is stateless.
    """
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4, seed=config.RANDOM_STATE)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="gtsrb_cnn")


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Adam + sparse categorical cross-entropy (labels are integers)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


def make_callbacks(checkpoint_path) -> list:
    """ModelCheckpoint (best val_accuracy) + EarlyStopping + ReduceLROnPlateau."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1),
    ]
