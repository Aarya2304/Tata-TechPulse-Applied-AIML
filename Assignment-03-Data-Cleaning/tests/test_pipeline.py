"""End-to-end test: run the full main() pipeline on the real dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.load_data import load_raw
from src.main import main


@pytest.fixture(scope="module")
def report() -> dict:
    return main()


class TestMainPipeline:
    def test_main_returns_report(self, report):
        assert "original_row_count" in report
        assert report["original_row_count"] == 8128

    def test_reports_written(self, report):
        assert config.DATA_PROFILE_PATH.exists()
        assert config.CLEANING_REPORT_PATH.exists()
        assert config.MISSING_REPORT_CSV_PATH.exists()
        assert config.OUTLIER_REPORT_CSV_PATH.exists()
        assert config.LEAKAGE_REPORT_PATH.exists()

    def test_plots_written(self, report):
        for name in ("missing_values_before.png", "missing_values_after.png",
                     "outliers_before_after.png", "scaling_comparison.png"):
            assert (config.PLOTS_DIR / name).exists()

    def test_processed_data_written_and_complete(self, report):
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        assert df.shape[0] == report["final_row_count"]
        # No NaNs anywhere in the processed feature matrix.
        assert not df.isna().any().any()
        # Raw target column preserved for reference.
        assert "selling_price_raw" in df.columns

    def test_duplicates_removed(self, report):
        assert report["duplicate_rows_removed"] == 1221
        assert report["rows_after_dedup"] == \
            report["original_row_count"] - 1221

    def test_missing_values_reduced_to_zero(self, report):
        assert report["missing_values_before_total"] > 0
        assert report["missing_values_after_total"] == 0

    def test_outliers_reported_and_capped(self, report):
        assert sum(report["outliers_detected"].values()) > 0
        assert sum(report["outliers_capped"].values()) > 0

    def test_leakage_verification_passes(self, report):
        lev = report["leakage_verification"]
        assert lev["imputer_medians_equal_train_medians"] is True
        assert lev["scaler_means_equal_capped_train_means"] is True
        assert lev["scaler_stats_unchanged_after_test_transform"] is True

    def test_profile_json_content(self, report):
        profile = json.loads(config.DATA_PROFILE_PATH.read_text())
        assert profile["n_rows"] == 8128
        assert profile["n_columns"] == 13
        assert profile["per_column"]["mileage"]["missing_count"] == 221

    def test_reproducible_processed_output(self, report):
        """Second full run must produce an identical processed CSV."""
        df1 = pd.read_csv(config.PROCESSED_DATA_PATH)
        report2 = main()
        df2 = pd.read_csv(config.PROCESSED_DATA_PATH)
        pd.testing.assert_frame_equal(df1, df2)
        assert report2["final_row_count"] == report["final_row_count"]
