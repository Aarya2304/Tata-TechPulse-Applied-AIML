"""Tests for text preprocessing (cleaning, splits, tokenizer, padding)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import (
    build_sequences,
    clean_text,
    compute_class_weights,
    fit_tokenizer,
    load_tokenizer,
    make_splits,
    save_tokenizer,
)


def test_clean_text_lowercases_and_strips():
    assert clean_text("  Great  PRODUCT!! ") == "great product!!"


def test_clean_text_preserves_negation_words():
    """Negation must survive cleaning — it carries sentiment."""
    t = clean_text("This is NOT good. Never buying again.")
    assert "not" in t.split() and "never" in t.split()


def test_clean_text_removes_urls_and_html():
    t = clean_text("See http://spam.example.com and <b>bold</b> claim")
    assert "http" not in t and "spam.example.com" not in t
    assert "bold" in t and "<b>" not in t


def test_make_splits_sizes_and_stratification():
    df = pd.DataFrame({
        config.TEXT_COLUMN: [f"text {i}" for i in range(1000)],
        config.LABEL_COLUMN: [1] * 800 + [0] * 200,
    })
    splits = make_splits(df, random_state=42)
    total = len(df)
    assert len(splits.train) == round(total * 0.8)
    assert len(splits.val) == round(total * 0.1)
    assert len(splits.test) == round(total * 0.1)
    # stratified: positive share preserved in every split (within rounding)
    for part in (splits.train, splits.val, splits.test):
        share = part[config.LABEL_COLUMN].mean()
        assert 0.78 <= share <= 0.82


def test_make_splits_deterministic():
    df = pd.DataFrame({
        config.TEXT_COLUMN: [f"t{i}" for i in range(200)],
        config.LABEL_COLUMN: [i % 2 for i in range(200)],
    })
    s1, s2 = make_splits(df), make_splits(df)
    assert s1.test[config.TEXT_COLUMN].tolist() == s2.test[config.TEXT_COLUMN].tolist()


def test_tokenizer_fit_and_oov():
    tokenizer = fit_tokenizer(["good car battery", "bad battery life"])
    seqs = tokenizer.texts_to_sequences(["good battery", "unknownword battery"])
    assert seqs[0] and seqs[1]
    assert 1 in seqs[1]  # <OOV> index for the unseen word


def test_tokenizer_is_fitted_only_on_training_data():
    """Words that appear ONLY in val/test must not be in the vocabulary."""
    train = ["good car", "bad car"]
    val = ["uniquevalidationword bad"]
    tok = fit_tokenizer(train)
    assert "uniquevalidationword" not in tok.word_index
    # converting val text still works via <OOV>
    seq = tok.texts_to_sequences(val)[0]
    assert 1 in seq


def test_build_sequences_shapes_and_padding():
    tok = fit_tokenizer(["short text", "a much longer review text here"])
    texts_a = ["short text", "a much longer review text here"]
    texts_b = ["short text"]
    texts_c = ["a much longer review text here"]
    seq = build_sequences(
        tok, texts_a, texts_b, texts_c,
        np.array([1, 0]), np.array([1]), np.array([0]),
    )
    expected_len = config.SEQUENCE_LENGTH
    assert seq.X_train.shape == (2, expected_len)
    assert seq.X_val.shape == (1, expected_len)
    assert seq.X_test.shape == (1, expected_len)
    # pre-padding: leading zeros, content ends at the sequence end (the
    # standard Keras recipe for recurrent models)
    row = seq.X_val[0]
    n_real = (row != 0).sum()
    assert (row[:len(row) - n_real] == 0).all()
    assert (row[len(row) - n_real:] != 0).all()


def test_vocab_size_capped_by_config():
    long_text = " ".join(f"w{i}" for i in range(50))
    tok = fit_tokenizer([long_text])
    seq = build_sequences(
        tok, [long_text], ["w0"], ["w1"],
        np.array([1]), np.array([1]), np.array([0]),
    )
    assert seq.vocab_size <= config.VOCAB_SIZE


def test_class_weights_balanced_rule():
    y = np.array([1] * 90 + [0] * 10)
    w = compute_class_weights(y)
    assert w[0] == pytest.approx(5.0)   # 100 / (2*10)
    assert w[1] == pytest.approx(100 / (2 * 90))


def test_tokenizer_json_roundtrip(tmp_path):
    tok = fit_tokenizer(["good car", "bad battery"])
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, path)
    tok2 = load_tokenizer(path)
    assert tok2.word_index == tok.word_index
    s1 = tok.texts_to_sequences(["good battery"])
    s2 = tok2.texts_to_sequences(["good battery"])
    assert s1 == s2
