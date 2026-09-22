"""Tests for src/cleaning.py (parsing, domain checks, outliers, capping)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.cleaning import (IQRCapper, apply_domain_checks,
                          count_capped_values, detect_outliers_iqr,
                          drop_exact_duplicates, iqr_bounds,
                          impute_dataframe, missing_before_after,
                          parse_features, parse_numeric_value, parse_torque)
from src.load_data import load_raw


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="module")
def parsed() -> pd.DataFrame:
    return parse_features(load_raw())


class TestLoading:
    def test_loads_successfully(self, raw):
        assert raw.shape[0] > 5000
        assert raw.shape[1] == 13

    def test_expected_columns_exist(self, raw):
        for col in config.RAW_COLUMNS:
            assert col in raw.columns


class TestParsers:
    def test_parse_numeric_value(self):
        assert parse_numeric_value("23.4 kmpl") == 23.4
        assert parse_numeric_value("1248 CC") == 1248.0
        assert parse_numeric_value("74 bhp") == 74.0
        assert parse_numeric_value(np.nan) is np.nan or np.isnan(
            parse_numeric_value(np.nan))

    def test_parse_torque_nm(self):
        assert parse_torque("190Nm@ 2000rpm") == pytest.approx(190.0)

    def test_parse_torque_kgm(self):
        assert parse_torque("22.4 kgm at 1750-2750rpm") == \
            pytest.approx(22.4 * 9.80665)

    def test_parse_torque_kgm_parenthesis(self):
        assert parse_torque("12.7@ 2,700(kgm@ rpm)") == \
            pytest.approx(12.7 * 9.80665)

    def test_parse_torque_nan(self):
        assert np.isnan(parse_torque(np.nan))

    def test_parsed_columns_added(self, parsed):
        for dst, _ in config.PARSED_NUM_COLUMNS.values():
            assert dst in parsed.columns
        assert "brand" in parsed.columns

    def test_parsed_engine_units(self, parsed):
        eng = parsed["engine_cc"].dropna()
        assert eng.between(600, 4000).all()

    def test_brand_extracted(self, parsed):
        assert parsed["brand"].nunique() >= 20
        assert parsed["brand"].iloc[0] == "Maruti"


class TestDuplicates:
    def test_duplicates_removed(self, parsed):
        deduped, n = drop_exact_duplicates(parsed)
        assert n > 1000            # dataset really has ~1202 dupes
        assert len(deduped) == len(parsed) - n
        assert deduped.duplicated().sum() == 0


class TestDomainChecks:
    def test_impossible_km_set_to_nan(self, parsed):
        deduped, _ = drop_exact_duplicates(parsed)
        checked, counts = apply_domain_checks(deduped)
        assert counts["km_driven_below_min"] >= 1
        assert checked["km_driven"].isna().sum() >= 1
        assert (checked["km_driven"].dropna()
                < config.MIN_PLAUSIBLE_KM_DRIVEN).sum() == 0

    def test_zero_mileage_and_power_set_to_nan(self, parsed):
        deduped, _ = drop_exact_duplicates(parsed)
        checked, counts = apply_domain_checks(deduped)
        assert (checked["mileage_kmpl"] == 0).sum() == 0
        assert (checked["max_power_bhp"] == 0).sum() == 0
        assert counts["mileage_kmpl"] >= 1
        assert counts["max_power_bhp"] >= 1


class TestIQR:
    def test_iqr_bounds(self):
        s = pd.Series(range(1, 101), dtype=float)  # Q1=25.75 Q3=75.25
        q1, q3, iqr, lo, hi = iqr_bounds(s)
        assert q1 == pytest.approx(25.75)
        assert q3 == pytest.approx(75.25)
        assert hi < 1e9

    def test_detect_outliers_flags_extremes(self):
        s = pd.concat([pd.Series(range(100), dtype=float),
                       pd.Series([1e6, -1e6])], ignore_index=True)
        df = pd.DataFrame({"x": s})
        report, bounds = detect_outliers_iqr(df, ["x"])
        assert report.loc[0, "outlier_count"] == 2

    def test_detect_on_real_data(self, parsed):
        deduped, _ = drop_exact_duplicates(parsed)
        report, _ = detect_outliers_iqr(deduped, ["km_driven",
                                                  "selling_price"])
        assert (report["outlier_count"] > 0).all()

    def test_report_percentages_consistent(self, parsed):
        deduped, _ = drop_exact_duplicates(parsed)
        report, _ = detect_outliers_iqr(deduped, ["km_driven"])
        pct = report.loc[0, "outlier_pct"]
        cnt = report.loc[0, "outlier_count"]
        n = deduped["km_driven"].notna().sum()
        assert pct == pytest.approx(100 * cnt / n, abs=0.01)


class TestIQRCapper:
    def test_capping_clips_extremes(self):
        df = pd.DataFrame({"x": [1.0] * 20 + [1000.0, -1000.0]})
        capped = IQRCapper().fit_transform(df)
        assert capped["x"].max() < 1000.0
        assert capped["x"].min() > -1000.0
        # Non-extreme values unchanged.
        assert (capped["x"].iloc[:20] == 1.0).all()

    def test_fit_on_train_only_transforms_test(self):
        train = pd.DataFrame({"x": np.random.RandomState(0).normal(0, 1, 500)})
        test = pd.DataFrame({"x": [50.0, -50.0]})
        capper = IQRCapper().fit(train)
        out = capper.transform(test)
        assert out["x"].max() <= capper.upper_[0] + 1e-9
        assert out["x"].min() >= capper.lower_[0] - 1e-9

    def test_capped_count_reported(self):
        df = pd.DataFrame({"x": [0.0] * 20 + [1e5]})
        capped = IQRCapper().fit_transform(df)
        n = count_capped_values(df, capped)["x"]
        assert n == 1


class TestImputation:
    def test_numeric_median_imputation(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = impute_dataframe(df, ["x"], [])
        assert out["x"].isna().sum() == 0
        assert out["x"].iloc[1] == 2.0

    def test_categorical_mode_imputation(self):
        df = pd.DataFrame({"c": ["a", "b", "a", np.nan]})
        out = impute_dataframe(df, [], ["c"])
        assert out["c"].isna().sum() == 0
        assert out["c"].iloc[3] == "a"

    def test_missing_before_after_table(self):
        before = pd.DataFrame({"x": [1.0, np.nan]})
        after = pd.DataFrame({"x": [1.0, 1.0]})
        table = missing_before_after(before, after)
        row = table[table.column == "x"].iloc[0]
        assert row.missing_before == 1 and row.missing_after == 0
