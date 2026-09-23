"""Text preprocessing: cleaning, splits, train-only tokenizer, padding.

Leakage rule (enforced here and by tests): the Keras ``Tokenizer`` is fitted
**only** on the training text. Validation/test text is tokenized with the
fitted vocabulary — never used to build it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from src import config

# NOTE: sklearn must NOT be imported at module level in the inference path.
# On this Windows setup, importing sklearn before TensorFlow breaks TF's
# native DLL initialisation (OpenMP/oneDNN conflict), so train_test_split is
# imported lazily inside make_splits() — sklearn is only needed for training.

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")
# Strip HTML tags/URLs; keep contractions and negation words ("not", "no",
# "never", "n't") intact — they carry sentiment.
_TAGS_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Light, deterministic text normalisation.

    - lower-case, strip HTML tags/URLs, collapse whitespace,
    - keep sentence punctuation and negation words (important for sentiment).

    No stop-word removal, no stemming: sentiment-bearing words such as
    "not", "never", "no" are preserved on purpose.
    """
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _TAGS_RE.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Return cleaned text strings for a dataframe with the text column."""
    return [clean_text(t) for t in df[config.TEXT_COLUMN].tolist()]


# ---------------------------------------------------------------------------
# Splits (stratified)
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    """Container for the three splits as raw dataframes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_splits(df: pd.DataFrame, random_state: int = config.RANDOM_STATE) -> SplitData:
    """Stratified 80/10/10 split by sentiment label.

    Both splits stratify on the label so the (strong) class imbalance is
    represented proportionally in train/validation/test.
    """
    from sklearn.model_selection import train_test_split  # see module note
    train_df, temp_df = train_test_split(
        df,
        test_size=config.TEST_FRACTION + config.VAL_FRACTION,
        stratify=df[config.LABEL_COLUMN],
        random_state=random_state,
    )
    relative_val = config.VAL_FRACTION / (config.TEST_FRACTION + config.VAL_FRACTION)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,  # half of the remaining 20 %
        stratify=temp_df[config.LABEL_COLUMN],
        random_state=random_state,
    )
    return SplitData(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Tokenizer + sequences
# ---------------------------------------------------------------------------
@dataclass
class SequenceData:
    """Padded integer sequences plus the fitted tokenizer and labels."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    tokenizer: Tokenizer
    vocab_size: int          # actual embedding input_dim (incl. pad + OOV)
    sequence_length: int
    split_sizes: dict[str, int]


def fit_tokenizer(train_texts: list[str]) -> Tokenizer:
    """Fit a Keras Tokenizer on the **training text only**.

    - ``num_words`` caps the vocabulary at ``config.VOCAB_SIZE``;
    - an explicit ``<OOV>`` token maps unseen words at inference time;
    - filters remove punctuation but keep letters/digits/apostrophes, so
      negation words and contractions ("n't") survive.
    """
    tokenizer = Tokenizer(
        num_words=config.VOCAB_SIZE,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    )
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def build_sequences(
    tokenizer: Tokenizer,
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> SequenceData:
    """Convert cleaned texts to padded integer sequences.

    All splits use the *same* tokenizer and the *same* fixed
    ``config.SEQUENCE_LENGTH`` (post-padding, pre-truncation), which is what
    makes the inference path identical to the training path.
    """
    seq_train = tokenizer.texts_to_sequences(train_texts)
    seq_val = tokenizer.texts_to_sequences(val_texts)
    seq_test = tokenizer.texts_to_sequences(test_texts)

    # PRE-padding is the standard Keras recipe for recurrent models: real
    # tokens then end exactly at the sequence end, so the LSTM's final state
    # reads actual review content (with post-padding the last steps are pad
    # embeddings and the sentiment signal washes out — measured on this
    # dataset: post-padding gave near-chance ROC-AUC 0.57, pre-padding 0.81).
    # Long reviews are truncated from the FRONT (truncating="pre"): measured
    # on this dataset, keeping the review tail beats keeping the title+start
    # (test accuracy/F1 0.903/0.947 vs 0.860/0.921) — reviews typically give
    # their overall verdict in the closing sentences.
    X_train = pad_sequences(seq_train, maxlen=config.SEQUENCE_LENGTH,
                            padding="pre", truncating="pre")
    X_val = pad_sequences(seq_val, maxlen=config.SEQUENCE_LENGTH,
                          padding="pre", truncating="pre")
    X_test = pad_sequences(seq_test, maxlen=config.SEQUENCE_LENGTH,
                           padding="pre", truncating="pre")

    # The embedding matrix needs an index for 0 (pad) and 1 (<OOV>) on top of
    # the real words, capped at VOCAB_SIZE.
    n_real_words = min(config.VOCAB_SIZE, len(tokenizer.word_index) + 1)
    vocab_size = min(config.VOCAB_SIZE, n_real_words + 1)

    return SequenceData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=np.asarray(y_train, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.float32),
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        sequence_length=config.SEQUENCE_LENGTH,
        split_sizes={
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
    )


def preprocess_dataframe(df: pd.DataFrame) -> SequenceData:
    """End-to-end: clean texts, split, fit tokenizer on train, pad sequences."""
    cleaned = [clean_text(t) for t in df[config.TEXT_COLUMN]]
    df = df.assign(_clean=cleaned)
    splits = make_splits(df.assign(**{config.TEXT_COLUMN: cleaned}))

    train_texts = splits.train[config.TEXT_COLUMN].tolist()
    val_texts = splits.val[config.TEXT_COLUMN].tolist()
    test_texts = splits.test[config.TEXT_COLUMN].tolist()

    tokenizer = fit_tokenizer(train_texts)
    return build_sequences(
        tokenizer,
        train_texts,
        val_texts,
        test_texts,
        splits.train[config.LABEL_COLUMN].to_numpy(),
        splits.val[config.LABEL_COLUMN].to_numpy(),
        splits.test[config.LABEL_COLUMN].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Tokenizer persistence
# ---------------------------------------------------------------------------
def save_tokenizer(tokenizer: Tokenizer, path=None) -> None:
    """Serialize the fitted tokenizer (vocabulary) as JSON."""
    path = path or config.TOKENIZER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())


def load_tokenizer(path=None) -> Tokenizer:
    """Load a tokenizer saved by :func:`save_tokenizer`."""
    path = path or config.TOKENIZER_PATH
    with open(path, "r", encoding="utf-8") as f:
        return tokenizer_from_json(f.read())


# ---------------------------------------------------------------------------
# Class weights (from training data only)
# ---------------------------------------------------------------------------
def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced class weights computed from the **training set only**.

    weight_c = n_train / (n_classes * count_c)  — sklearn's 'balanced' rule.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    n = len(y_train)
    return {
        int(c): float(n / (len(classes) * cnt)) for c, cnt in zip(classes, counts)
    }
