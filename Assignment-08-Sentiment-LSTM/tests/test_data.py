"""Tests for the data loader (synthetic frames + real dataset when present)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import clean_dataframe


def _frame(texts, ratings, titles=None):
    titles = titles or [""] * len(texts)
    return pd.DataFrame(
        {config.TEXT_COLUMN: texts, config.TITLE_COLUMN: titles,
         config.RATING_COLUMN: ratings}
    )


def test_clean_dataframe_returns_expected_columns():
    df = _frame(["Great product for my car", "Terrible quality"], [5, 1])
    out, stats = clean_dataframe(df)
    assert list(out.columns) == [config.TEXT_COLUMN, config.RATING_COLUMN, config.LABEL_COLUMN]
    assert len(out) == 2
    assert set(out[config.LABEL_COLUMN]) == {0, 1}


def test_label_derivation_from_ratings():
    df = _frame(["a", "b", "c", "d", "e"], [1, 2, 3, 4, 5])
    out, stats = clean_dataframe(df)
    labels = dict(zip(out[config.RATING_COLUMN], out[config.LABEL_COLUMN]))
    assert labels[1.0] == 0 and labels[2.0] == 0      # negative
    assert 3.0 not in labels                          # neutral dropped
    assert labels[4.0] == 1 and labels[5.0] == 1      # positive
    assert stats["neutral_reviews_dropped"] == 1


def test_missing_text_and_rating_dropped():
    df = _frame(["good", None, "bad", ""], [5, 1, None, 2])
    out, stats = clean_dataframe(df)
    assert len(out) == 1                    # only the complete row survives
    assert out.iloc[0][config.LABEL_COLUMN] == 1
    assert stats["rows_dropped_missing_text_or_label"] >= 1


def test_duplicates_removed():
    df = _frame(["same review text"] * 3 + ["unique review"], [5, 5, 5, 1])
    out, stats = clean_dataframe(df)
    assert stats["duplicates_removed"] == 2
    assert len(out) == 2


def test_title_prepended_to_text():
    df = _frame(["the battery works"], [5], ["Excellent"])
    out, _ = clean_dataframe(df)
    assert "excellent" in out.iloc[0][config.TEXT_COLUMN].lower()


def test_stats_are_measured_not_hardcoded():
    df = _frame(["a good part"], [4])
    _, stats = clean_dataframe(df)
    assert stats["rows_final"] == len(_frame(["a good part"], [4]))
    assert stats["class_counts"]["1"] == 1
    assert abs(sum(stats["class_distribution"].values()) - 1.0) < 1e-6


def test_real_dataset_loads_when_present():
    """Full loader on the real archive when it has been downloaded."""
    if not config.RAW_DATA_PATH.exists():
        pytest.skip("dataset not downloaded yet (run: python -m src.train)")
    from src.data_loader import load_and_clean

    df, stats = load_and_clean()
    assert stats["rows_loaded"] == 20473
    assert stats["rows_final"] == 19032
    assert stats["duplicates_removed"] == 5
    assert stats["neutral_reviews_dropped"] == 1430
    assert stats["class_counts"] == {"0": 1147, "1": 17885}
    assert set(df[config.LABEL_COLUMN].unique()) == {0, 1}
