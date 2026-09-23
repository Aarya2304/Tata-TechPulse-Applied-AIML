"""End-to-end prediction-pipeline tests (tiny model + real preprocessing)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from tensorflow.keras.models import load_model

from src import config
from src.model import build_lstm_model
from src.preprocessing import fit_tokenizer, load_tokenizer, save_tokenizer
from src.predict import predict_sentiment
from src.train_utils import pad_texts


@pytest.fixture(scope="module")
def tiny_artifacts(tmp_path_factory):
    """Train a tiny LSTM on synthetic sentiment-ish sentences and save it."""
    tmp = tmp_path_factory.mktemp("artifacts")
    pos = ["great product for my car", "excellent battery works well",
           "love this item very good", "works perfectly on my truck"]
    neg = ["terrible quality broke fast", "bad product not good at all",
           "would never buy this again", "worst purchase ever made"]
    texts = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.float32)

    tok = fit_tokenizer(texts)
    X = pad_texts(texts, tok, {"sequence_length": 16})

    model = build_lstm_model(vocab_size=64, sequence_length=16,
                             embedding_dim=8, lstm_units=8, dense_units=4,
                             dropout=0.1)
    model.fit(X, labels, epochs=30, batch_size=4, verbose=0)

    model_path = tmp / "model.keras"
    tok_path = tmp / "tokenizer.json"
    meta_path = tmp / "metadata.json"
    model.save(model_path)
    save_tokenizer(tok, tok_path)
    meta = {"sequence_length": 16, "class_mapping": {"0": "Negative", "1": "Positive"}}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return model_path, tok_path, meta_path


def test_saved_model_loads(tiny_artifacts):
    model_path, _, _ = tiny_artifacts
    model = load_model(model_path)
    assert model is not None


def test_tokenizer_reload_matches_training_vocab(tiny_artifacts):
    _, tok_path, _ = tiny_artifacts
    tok = load_tokenizer(tok_path)
    assert "car" in tok.word_index and "terrible" in tok.word_index


def test_metadata_loads(tiny_artifacts):
    _, _, meta_path = tiny_artifacts
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["sequence_length"] == 16


def test_pad_texts_uses_saved_sequence_length(tiny_artifacts):
    _, tok_path, meta_path = tiny_artifacts
    tok = load_tokenizer(tok_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    X = pad_texts(["good car"], tok, meta)
    assert X.shape == (1, meta["sequence_length"])


def test_prediction_pipeline_positive_and_negative(tiny_artifacts):
    model_path, tok_path, meta_path = tiny_artifacts
    model = load_model(model_path)
    tok = load_tokenizer(tok_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    pos = predict_sentiment("great product for my car", model, tok, meta)
    neg = predict_sentiment("terrible quality broke fast", model, tok, meta)

    # The toy model is trained for seconds, so we assert the meaningful
    # property (positive sentence ranks above negative one) plus valid
    # output structure, not a specific 0.5-boundary decision.
    assert pos["probability"] > neg["probability"]
    assert pos["sentiment"] in ("Positive", "Negative")
    assert neg["sentiment"] in ("Positive", "Negative")
    for r in (pos, neg):
        assert r["label"] in (0, 1)
        assert 0.0 <= r["probability"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0


def test_prediction_handles_unseen_words(tiny_artifacts):
    """Fully unseen vocabulary must still produce a valid probability."""
    model_path, tok_path, meta_path = tiny_artifacts
    model = load_model(model_path)
    tok = load_tokenizer(tok_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    result = predict_sentiment("zzzqqq xyzzy flurb", model, tok, meta)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["sentiment"] in ("Positive", "Negative")
