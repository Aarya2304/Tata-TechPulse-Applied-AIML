"""Tests for data loading, cleaning and target-exclusion (leakage guard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (apply_domain_checks, drop_exact_duplicates,
                             load_raw)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_dataset_loads(raw):
    assert len(raw) == 8128
    assert list(raw.columns) == config.RAW_COLUMNS


def test_target_exists(raw):
    assert config.TARGET == "selling_price"
    assert config.TARGET in raw.columns
    assert raw[config.TARGET].notna().all()


def test_target_is_positive(raw):
    assert (raw[config.TARGET] > 0).all()


def test_duplicate_rows_removed(raw):
    deduped, n_removed = drop_exact_duplicates(raw)
    assert n_removed == 1202          # real measured duplicate count (raw)
    assert len(deduped) == len(raw) - n_removed
    assert not deduped.duplicated().any()


def test_domain_checks_zero_mileage():
    df = pd.DataFrame({"km_driven": [50_000.0, 42.0, 10_000.0],
                       "mileage_kmpl": [18.0, 15.0, 0.0],
                       "max_power_bhp": [88.0, 0.0, 100.0]})
    out, n_fixed = apply_domain_checks(df)
    assert n_fixed == 3               # km=42, mileage=0, bhp=0
    assert np.isnan(out.loc[1, "km_driven"])
    assert np.isnan(out.loc[2, "mileage_kmpl"])
    assert np.isnan(out.loc[1, "max_power_bhp"])
    assert out.loc[0].notna().all()   # plausible row untouched


def test_domain_checks_do_not_touch_plausible_values(raw):
    from src.feature_engineering import parse_features
    parsed = parse_features(raw)
    out, _ = apply_domain_checks(parsed)
    plausible = parsed[parsed["km_driven"] >= 100]
    assert (out.loc[plausible.index, "km_driven"]
            == plausible["km_driven"]).all()


def test_feature_selection_excludes_target():
    """The explicit feature list used for model inputs must not contain the
    target or any column derived from it."""
    assert config.TARGET not in config.FEATURES
    assert "selling_price" not in config.NUMERIC_FEATURES
    assert "selling_price" not in config.CATEGORICAL_FEATURES
    assert "year" not in config.FEATURES        # replaced by vehicle_age
    assert "name" not in config.FEATURES        # replaced by brand
