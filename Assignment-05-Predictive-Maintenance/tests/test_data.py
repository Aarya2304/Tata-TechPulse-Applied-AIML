"""Tests for data loading, validation and target properties."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import (drop_duplicates, load_and_prepare, load_raw,
                             target_distribution, validate)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_dataset_loads(raw):
    assert len(raw) == 10000
    assert list(raw.columns) == config.RAW_COLUMNS


def test_required_columns_exist(raw):
    missing = [c for c in config.RAW_COLUMNS if c not in raw.columns]
    assert missing == []


def test_validate_accepts_real_data(raw):
    validate(raw)          # must not raise


def test_validate_rejects_bad_target():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"Machine failure": [0, 2]}))


def test_target_is_binary(raw):
    assert set(raw[config.TARGET].unique()) == {0, 1}


def test_target_distribution_matches_documented_values(raw):
    dist = target_distribution(raw)
    assert dist[0] == 9661            # measured: no-failure rows
    assert dist[1] == 339             # measured: failure rows


def test_class_imbalance_is_strong(raw):
    """Documents the 28.5:1 imbalance that motivates imbalance-aware metrics."""
    dist = target_distribution(raw)
    ratio = dist[0] / dist[1]
    assert 25 < ratio < 32


def test_no_missing_values(raw):
    assert raw.isna().sum().sum() == 0


def test_no_exact_duplicates(raw):
    assert raw.duplicated().sum() == 0
    deduped, removed = drop_duplicates(raw)
    assert removed == 0
    assert len(deduped) == len(raw)


def test_leakage_columns_are_target_components(raw):
    """Failure-mode flags must be rare and dominated by failure rows."""
    for col in config.LEAKAGE_COLUMNS:
        assert set(raw[col].unique()) <= {0, 1}
        assert raw[col].sum() < 200   # all five are rare events
    # almost every flagged row is a failure row (the 18 exceptions are
    # documented in the README)
    flagged = raw[raw[config.LEAKAGE_COLUMNS].sum(axis=1) > 0]
    assert (flagged[config.TARGET] == 1).mean() > 0.9


def test_load_and_prepare_pipeline():
    df = load_and_prepare(verbose=False)
    assert len(df) == 10000
    assert config.TARGET in df.columns
