"""Shared helpers for the training and inference paths (kept dependency-light)."""

from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import clean_text


def pad_texts(texts: list[str], tokenizer, metadata: dict) -> np.ndarray:
    """Convert raw sentences to padded sequences with **saved** settings.

    Uses the saved fitted tokenizer and the saved sequence length, so the
    inference path cannot drift from the training path.
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    return pad_sequences(
        sequences,
        maxlen=int(metadata["sequence_length"]),
        padding="pre",
        truncating="post",
    )
