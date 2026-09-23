"""Tests for data loading, parsing, feature engineering and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (
    FEATURES,
    engineer_features,
    load_and_prepare,
    make_splits,
    parse_engine,
    parse_max_power,
    parse_mileage,
    parse_torque,
)

TARGET = config.TARGET


@pytest.fixture(scope="module")
def prepared():
    X, y, provenance = load_and_prepare()
    return X, y, provenance


def test_dataset_loads(prepared):
    X, y, provenance = prepared
    assert len(X) > 7000
    assert provenance["rows_raw"] == 8128
    assert provenance["rows_final"] == 8128   # no missing targets in this file
    assert provenance["rows_dropped_missing_target"] == 0


def test_engineered_columns_exist(prepared):
    X, _, _ = prepared
    for col in FEATURES:
        assert col in X.columns, col


def test_parsers_handle_real_formats():
    assert parse_mileage(pd.Series(["23.4 kmpl", "21.14 km/kg", None])).tolist()[:2] == [23.4, 21.14]
    assert parse_engine(pd.Series(["1248 CC", "998 CC"])).tolist() == [1248.0, 998.0]
    assert parse_max_power(pd.Series(["74 bhp", "103.52 bhp"])).tolist() == [74.0, 103.52]
    # kgm must convert to Nm (x 9.80665)
    torque = parse_torque(pd.Series(["190Nm@ 2000rpm", "12.7 kgm@ 2500 rpm", None]))
    assert torque.iloc[0] == pytest.approx(190.0)
    assert torque.iloc[1] == pytest.approx(12.7 * 9.80665)
    assert np.isnan(torque.iloc[2])


def test_vehicle_age_and_km_per_year(prepared):
    X, _, _ = prepared
    assert (X["vehicle_age"] == 2020 - X["year"]).all()
    assert (X["vehicle_age"] >= 0).all()
    # km_per_year is NaN only where age == 0 (2020 models); otherwise it is
    # bounded by km_driven (age >= 1 year)
    valid = X["km_per_year"].notna()
    assert (valid == (X["vehicle_age"] > 0)).all()
    assert (X.loc[valid, "km_per_year"] <= X.loc[valid, "km_driven"]).all()


def test_target_exists_and_is_numeric(prepared):
    _, y, _ = prepared
    assert y.name == TARGET
    assert pd.api.types.is_numeric_dtype(y)
    assert y.notna().all()


def test_no_target_leakage_in_features(prepared):
    X, _, _ = prepared
    assert TARGET not in X.columns
    assert "name" not in X.columns          # identifier replaced by brand
    assert "torque" not in X.columns        # raw string replaced by torque_nm
    assert not any("selling" in c.lower() for c in X.columns)


def test_split_is_reproducible(prepared):
    X, y, _ = prepared
    Xa_tr, Xa_te, ya_tr, ya_te = make_splits(X, y, random_state=42)
    Xb_tr, Xb_te, yb_tr, yb_te = make_splits(X, y, random_state=42)
    pd.testing.assert_frame_equal(Xa_tr, Xb_tr)
    pd.testing.assert_series_equal(ya_te, yb_te)
    assert len(Xa_te) + len(Xa_tr) == len(X)
    assert abs(len(Xa_te) - len(X) * config.TEST_SIZE) <= 1
