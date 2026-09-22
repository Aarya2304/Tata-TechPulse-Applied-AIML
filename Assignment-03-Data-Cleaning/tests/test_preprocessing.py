"""Tests for src/preprocessing.py (pipelines, scalers, OHE, leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.cleaning import IQRCapper
from src.load_data import load_raw
from src.preprocessing import (PreprocessingBuilder, build_categorical_pipeline,
                               build_numeric_pipeline, leakage_verification,
                               scaling_comparison_stats)


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    """Small cleaned slice of the real data (fast, deterministic)."""
    from src.cleaning import (apply_domain_checks, drop_exact_duplicates,
                              impute_dataframe, parse_features)
    df = parse_features(load_raw())
    df, _ = drop_exact_duplicates(df)
    df, _ = apply_domain_checks(df)
    df = impute_dataframe(df, config.NUMERIC_FEATURES,
                          config.CATEGORICAL_FEATURES)
    return df


class TestNumericPipeline:
    def test_median_imputation_works(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0, 100.0]})
        pipe = build_numeric_pipeline("standard")
        pipe.fit(df)
        out = pipe.transform(df)
        assert not np.isnan(out).any()
        # median of [1,3,5,100] is 4
        assert np.isclose(pipe.named_steps["imputer"].statistics_[0], 4.0)

    def test_standard_scaler_output_stats(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"a": rng.normal(10, 3, 1000)})
        out = build_numeric_pipeline("standard").fit_transform(df)
        assert out.mean() == pytest.approx(0.0, abs=1e-7)
        assert out.std() == pytest.approx(1.0, abs=1e-3)

    def test_robust_scaler_output_stats(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"a": np.concatenate(
            [rng.normal(10, 3, 900), [500.0] * 100])})
        out = build_numeric_pipeline("robust").fit_transform(df)
        med = np.median(out)
        assert med == pytest.approx(0.0, abs=1e-7)

    def test_invalid_scaler_name_raises(self):
        with pytest.raises(ValueError):
            build_numeric_pipeline("quantum")

    def test_iqr_cap_inside_pipeline(self):
        df = pd.DataFrame({"a": [1.0] * 50 + [1e6]})
        out = build_numeric_pipeline("standard").fit_transform(df)
        # Without the cap the outlier would dominate; with it, bounded.
        assert np.abs(out).max() < 10


class TestCategoricalPipeline:
    def test_most_frequent_imputation(self):
        df = pd.DataFrame({"c": ["a", "b", "a", np.nan]})
        pipe = build_categorical_pipeline()
        pipe.fit(df)
        assert pipe.named_steps["imputer"].statistics_[0] == "a"

    def test_onehot_output_columns(self):
        df = pd.DataFrame({"c": ["a", "b", "a"]})
        out = build_categorical_pipeline().fit_transform(df)
        assert out.shape == (3, 2)

    def test_unknown_category_ignored(self):
        train = pd.DataFrame({"c": ["a", "b", "a", "b"]})
        pipe = build_categorical_pipeline().fit(train)
        test = pd.DataFrame({"c": ["unknown_brand", "a"]})
        out = pipe.transform(test)
        # Unknown row -> all-zero one-hot; no exception raised.
        assert not np.isnan(out).any()
        assert (out[0] == 0).all()


class TestColumnTransformer:
    def test_full_pipeline_fit_transform(self, cleaned):
        X = cleaned[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
        builder = PreprocessingBuilder(scaler="robust")
        out = builder.build().fit_transform(X)
        assert not np.isnan(np.asarray(out)).any()
        assert out.shape[0] == X.shape[0]
        names = builder.feature_names
        assert any(n.startswith("num__") for n in names)
        assert any(n.startswith("cat__") for n in names)

    def test_output_matches_feature_count(self, cleaned):
        X = cleaned[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
        builder = PreprocessingBuilder(scaler="standard")
        builder.build().fit(X)
        n_num = len(config.NUMERIC_FEATURES)
        n_cat = sum(cleaned[c].nunique() for c in config.CATEGORICAL_FEATURES)
        assert len(builder.feature_names) == n_num + n_cat

    def test_no_nan_after_preprocessing(self, cleaned):
        X = cleaned[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
        # Inject some NaNs to prove the pipeline fills them.
        X = X.copy()
        X.loc[X.index[:5], "km_driven"] = np.nan
        X.loc[X.index[:5], "fuel"] = None
        out = PreprocessingBuilder(scaler="standard").build().fit_transform(X)
        assert not np.isnan(np.asarray(out)).any()


class TestLeakage:
    def test_fit_on_train_only(self, cleaned):
        rng = np.random.RandomState(config.RANDOM_STATE)
        mask = rng.rand(len(cleaned)) < 0.25
        train, test = cleaned[~mask], cleaned[mask]
        report = leakage_verification(train, test)
        assert report["imputer_medians_equal_train_medians"] is True
        assert report["scaler_means_equal_capped_train_means"] is True
        assert report["scaler_stats_unchanged_after_test_transform"] is True

    def test_transform_does_not_mutate_fitted_state(self, cleaned):
        X = cleaned[config.NUMERIC_FEATURES]
        pipe = build_numeric_pipeline("standard").fit(X)
        mean_before = pipe.named_steps["scaler"].mean_.copy()
        pipe.transform(X.head(10))          # "test" data
        assert np.allclose(mean_before, pipe.named_steps["scaler"].mean_)


class TestScalingComparison:
    def test_comparison_stats_structure(self, cleaned):
        cols = config.SCALER_COMPARISON_FEATURES[:2]
        stats = scaling_comparison_stats(cleaned, cols)
        for col in cols:
            assert {"raw", "standard_scaled", "robust_scaled"} <= \
                set(stats[col])
            assert {"mean", "std", "median", "iqr"} <= \
                set(stats[col]["raw"])

    def test_standard_zero_mean_robust_zero_median(self, cleaned):
        col = "km_driven"
        stats = scaling_comparison_stats(cleaned, [col])
        assert stats[col]["standard_scaled"]["mean"] == pytest.approx(
            0.0, abs=1e-6)
        assert stats[col]["robust_scaled"]["median"] == pytest.approx(
            0.0, abs=1e-6)
