"""LSTM sentiment classifier (TensorFlow/Keras).

Architecture (CPU-friendly, documented in the README):

    token sequence
      → Embedding(vocab_size, EMBEDDING_DIM)   # learned word vectors
      → LSTM(LSTM_UNITS)                       # sequential context encoder
      → Dropout(DROPOUT)                       # regularisation
      → Dense(DENSE_UNITS, relu)               # feed-forward head
      → Dense(1, sigmoid)                      # P(positive sentiment)

No pretrained language model and no transformer component is used — the only
learned text representation is this network's own embedding layer.
"""

from __future__ import annotations

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src import config


def build_lstm_model(
    vocab_size: int = config.VOCAB_SIZE,
    sequence_length: int = config.SEQUENCE_LENGTH,
    embedding_dim: int = config.EMBEDDING_DIM,
    lstm_units: int = config.LSTM_UNITS,
    dense_units: int = config.DENSE_UNITS,
    dropout: float = config.DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> Sequential:
    """Construct and compile the LSTM sentiment classifier.

    Parameters mirror ``config.py`` so all hyper-parameters live in one place.
    """
    model = Sequential(
        [
            # Explicit input: integer token sequences of fixed length.
            Input(shape=(sequence_length,), dtype="int32", name="token_ids"),
            # Embedding: maps each token index to a dense vector of
            # ``embedding_dim`` floats, learned from scratch during training.
            # mask_zero=True makes the LSTM skip pad steps (index 0), so the
            # final state depends only on real review tokens.
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True,
            ),
            # LSTM: reads the sequence of word vectors and summarises it into
            # ``lstm_units`` features; the final state feeds the classifier.
            LSTM(lstm_units),
            # Dropout: randomly zeroes ``dropout`` fraction of the LSTM
            # features during training to reduce overfitting.
            Dropout(dropout),
            # Dense head: non-linear combination of the LSTM summary.
            Dense(dense_units, activation="relu"),
            # Output: single sigmoid unit = P(positive sentiment).
            Dense(1, activation="sigmoid"),
        ],
        name="sentiment_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks(model_path=None) -> list:
    """Standard callbacks: checkpoint best val_loss, stop early, decay LR."""
    model_path = model_path or config.MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor=config.MONITOR_METRIC,
            save_best_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor=config.MONITOR_METRIC,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=config.MONITOR_METRIC,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
