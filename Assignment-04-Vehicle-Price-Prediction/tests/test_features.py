"""Tests for unit parsing and engineered automotive features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import REFERENCE_YEAR
from src.feature_engineering import (add_km_per_year, add_vehicle_age,
                                     build_features, parse_features,
                                     parse_numeric_value, parse_torque)


def test_parse_mileage():
    assert parse_numeric_value("23.4 kmpl") == pytest.approx(23.4)


def test_parse_engine():
    assert parse_numeric_value("1248 CC") == pytest.approx(1248.0)


def test_parse_horsepower():
    assert parse_numeric_value("74 bhp") == pytest.approx(74.0)
    assert parse_numeric_value("68.05 bhp") == pytest.approx(68.05)


def test_parse_torque_formats():
    # three real formats from the dataset
    assert parse_torque("190Nm@ 2000rpm") == pytest.approx(190.0)
    assert parse_torque("22.4 kgm at 1750-2750rpm") == pytest.approx(
        22.4 * 9.80665)
    assert parse_torque("12.7@ 2,700(kgm@ rpm)") == pytest.approx(
        12.7 * 9.80665)


def test_parse_handles_missing():
    assert np.isnan(parse_numeric_value(np.nan))
    assert np.isnan(parse_numeric_value("no digits here"))
    assert np.isnan(parse_torque(np.nan))


def test_vehicle_age_calculation():
    df = pd.DataFrame({"year": [2020, 2010, 1995]})
    out = add_vehicle_age(df)
    assert list(out["vehicle_age"]) == [REFERENCE_YEAR - 2020,
                                        REFERENCE_YEAR - 2010,
                                        REFERENCE_YEAR - 1995]
    assert (out["vehicle_age"] >= 0).all()


def test_km_per_year():
    df = pd.DataFrame({"vehicle_age": [5, 0], "km_driven": [50_000, 10_000]})
    out = add_km_per_year(df)
    assert out.loc[0, "km_per_year"] == pytest.approx(10_000.0)
    # age 0 is clipped to 1 so km_per_year stays finite
    assert out.loc[1, "km_per_year"] == pytest.approx(10_000.0)


def test_brand_extraction():
    df = pd.DataFrame({"name": ["Maruti Swift VDI  ", "Hyundai i10 Magna"],
                       "year": [2017, 2015],
                       "mileage": ["23.4 kmpl", "19.77 kmpl"],
                       "engine": ["1248 CC", "998 CC"],
                       "max_power": ["74 bhp", "68.05 bhp"],
                       "torque": ["190Nm@ 2000rpm", "90Nm@ 3500rpm"]})
    out = parse_features(df)
    assert list(out["brand"]) == ["Maruti", "Hyundai"]
    assert "name" not in out.columns          # consumed
    assert out.loc[0, "mileage_kmpl"] == pytest.approx(23.4)


def test_build_features_adds_all_derived_columns():
    df = pd.DataFrame({
        "name": ["Maruti Swift VDI"], "year": [2017],
        "selling_price": [500_000], "km_driven": [45_000],
        "fuel": ["Diesel"], "seller_type": ["Individual"],
        "transmission": ["Manual"], "owner": ["First Owner"],
        "mileage": ["23.4 kmpl"], "engine": ["1248 CC"],
        "max_power": ["74 bhp"], "torque": ["190Nm@ 2000rpm"],
        "seats": [5.0],
    })
    out = build_features(df)
    for col in ["vehicle_age", "km_per_year", "brand", "mileage_kmpl",
                "engine_cc", "max_power_bhp", "torque_nm"]:
        assert col in out.columns
    assert out.loc[0, "vehicle_age"] == REFERENCE_YEAR - 2017
    # target passes through untouched
    assert out.loc[0, "selling_price"] == 500_000


def test_features_are_target_independent():
    """Engineered columns must be computable without the target present."""
    df = pd.DataFrame({
        "name": ["Tata Nexon XM"], "year": [2019], "km_driven": [30_000],
        "mileage": ["17.4 kmpl"], "engine": ["1497 CC"],
        "max_power": ["108 bhp"], "torque": ["260Nm@ 1500rpm"],
    })
    out = build_features(df)   # no selling_price column at all
    assert out[["vehicle_age", "brand", "torque_nm"]].notna().all().all()
