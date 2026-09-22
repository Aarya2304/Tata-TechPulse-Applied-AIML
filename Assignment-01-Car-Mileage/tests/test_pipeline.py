"""Automated tests for the Car Mileage assignment pipeline.

Run from the project root:

    python -m pytest tests -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config
from src.data_pipeline import (
    HorsepowerImputer,
    clean,
    load_and_clean,
    load_raw,
    missing_value_report,
)
from src.metrics_utils import cross_validate_model, regression_metrics
from src.train import build_models, make_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return load_and_clean()


@pytest.fixture(scope="session")
def split(clean_df) -> tuple:
    return make_split(clean_df)


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
class TestDataLoading:
    def test_raw_shape(self, raw_df):
        # The UCI Auto MPG dataset ships exactly 398 rows and 9 columns.
        assert raw_df.shape == (398, 9)
        assert list(raw_df.columns) == config.COLUMN_NAMES

    def test_missing_horsepower_detected(self, raw_df):
        # 6 rows carry the UCI '?' marker in horsepower.
        assert raw_df["horsepower"].isna().sum() == 6

    def test_clean_drops_no_valid_rows(self, raw_df, clean_df):
        # No duplicate rows and no missing targets in this dataset.
        assert len(clean_df) == len(raw_df)

    def test_no_missing_values_after_clean(self, clean_df):
        # Missing horsepower still present (imputation happens inside the
        # pipeline, leakage-free) but everything else must be complete.
        rest = clean_df.drop(columns=["horsepower"])
        assert rest.isna().sum().sum() == 0

    def test_missing_report(self, clean_df):
        report = missing_value_report(clean_df)
        assert list(report.index) == ["horsepower"]
        assert report.loc["horsepower", "missing_count"] == 6

    def test_dtypes_numeric(self, clean_df):
        for col in [config.TARGET] + config.FEATURES:
            assert pd.api.types.is_numeric_dtype(clean_df[col])

    def test_mpg_in_plausible_range(self, clean_df):
        assert clean_df[config.TARGET].between(5, 60).all()


# ---------------------------------------------------------------------------
# Missing-value handling (leakage-free imputer)
# ---------------------------------------------------------------------------
class TestHorsepowerImputer:
    def test_fit_learns_group_medians(self, clean_df):
        train_df, _ = make_split(clean_df)
        imp = HorsepowerImputer().fit(train_df)
        assert imp.overall_median_ > 0
        assert len(imp.group_medians_) == train_df["cylinders"].nunique()

    def test_transform_fills_all_missing(self, clean_df):
        train_df, test_df = make_split(clean_df)
        imp = HorsepowerImputer().fit(train_df)
        filled = imp.transform(test_df)
        assert filled["horsepower"].isna().sum() == 0
        # Same number of rows in, same number out.
        assert len(filled) == len(test_df)

    def test_no_leakage_train_stats_only(self, clean_df):
        """Imputer fitted on train must not change when test rows change."""
        train_df, test_df = make_split(clean_df)
        imp = HorsepowerImputer().fit(train_df)
        before = dict(imp.group_medians_)
        mutated_test = test_df.copy()
        mutated_test["horsepower"] = 9999.0  # poison the test split
        imp.transform(mutated_test)
        assert dict(imp.group_medians_) == before

    def test_fallback_for_unseen_cylinders(self, clean_df):
        train_df, _ = make_split(clean_df)
        imp = HorsepowerImputer().fit(train_df)
        weird = pd.DataFrame({
            "cylinders": [99], "horsepower": [np.nan],
            "displacement": [100.0], "weight": [2000.0],
        })
        filled = imp.transform(weird)
        assert filled["horsepower"].iloc[0] == imp.overall_median_

    def test_works_inside_sklearn_pipeline(self, clean_df):
        train_df, _ = make_split(clean_df)
        X = train_df[config.FEATURES]
        pipe = Pipeline([("imputer", HorsepowerImputer())]).fit(X)
        out = pipe.transform(X)
        assert isinstance(out, pd.DataFrame)
        assert out.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
class TestSplit:
    def test_split_sizes(self, clean_df, split):
        train_df, test_df = split
        n = len(train_df) + len(test_df)
        assert n == len(clean_df)
        assert abs(len(test_df) / n - config.TEST_SIZE) < 0.03

    def test_no_overlap(self, clean_df, split):
        train_df, test_df = split
        train_keys = set(map(tuple, train_df[config.FEATURES + [config.TARGET]].values))
        test_keys = set(map(tuple, test_df[config.FEATURES + [config.TARGET]].values))
        assert not (train_keys & test_keys)

    def test_reproducible(self, clean_df):
        t1, e1 = make_split(clean_df)
        t2, e2 = make_split(clean_df)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(e1, e2)

    def test_target_ranges_match(self, clean_df, split):
        """Both splits must span the same mpg range (stratification works)."""
        train_df, test_df = split
        assert train_df[config.TARGET].min() < 15
        assert test_df[config.TARGET].max() > 40


# ---------------------------------------------------------------------------
# Models & metrics
# ---------------------------------------------------------------------------
class TestModels:
    def test_all_three_models_exist(self):
        models = build_models()
        assert set(models) == {"Linear Regression", "Ridge Regression",
                               "Random Forest"}
        for name, model in models.items():
            assert isinstance(model, Pipeline), name

    def test_models_fit_and_predict(self, split):
        train_df, test_df = split
        X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]
        X_test = test_df[config.FEATURES]
        for name, model in build_models().items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            assert len(preds) == len(X_test)
            assert np.isfinite(preds).all()

    def test_metrics_reasonable(self, split):
        train_df, test_df = split
        X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]
        X_test, y_test = test_df[config.FEATURES], test_df[config.TARGET]
        model = build_models()["Random Forest"].fit(X_train, y_train)
        m = regression_metrics(y_test, model.predict(X_test))
        # Sanity bands: a working model on Auto MPG must clear these easily.
        assert m["R2"] > 0.70
        assert 0 < m["MAE"] < 5
        assert 0 < m["RMSE"] < 6

    def test_cv_returns_all_metrics(self, split):
        train_df = split[0]
        X, y = train_df[config.FEATURES], train_df[config.TARGET]
        cv = cross_validate_model(build_models()["Ridge Regression"], X, y)
        for metric in ("MAE", "RMSE", "R2"):
            assert metric in cv
            assert cv[metric]["std"] >= 0
        assert cv["R2"]["mean"] > 0.60


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_full_pipeline(self, clean_df):
        """Train every model, evaluate on a held-out split, verify quality."""
        train_df, test_df = make_split(clean_df)
        X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]
        X_test, y_test = test_df[config.FEATURES], test_df[config.TARGET]
        results = {}
        for name, model in build_models().items():
            model.fit(X_train, y_train)
            results[name] = regression_metrics(y_test, model.predict(X_test))
        # Every candidate must beat a naive mean-predictor (R2 > 0).
        assert all(r["R2"] > 0 for r in results.values())
        # And at least one model should be genuinely good.
        assert max(r["R2"] for r in results.values()) > 0.80
