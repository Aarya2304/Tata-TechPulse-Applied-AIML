"""Tests for the sklearn preprocessing (numeric + categorical pipelines)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.preprocessing import build_preprocessor

NUMERIC = config.NUMERIC_FEATURES
CATEGORICAL = config.CATEGORICAL_FEATURES


@pytest.fixture()
def sample_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "vehicle_age": rng.integers(0, 12, n).astype(float),
        "km_driven": rng.integers(5_000, 150_000, n).astype(float),
        "mileage_kmpl": np.concatenate([rng.normal(19, 3, n - 2), [np.nan] * 2]),
        "engine_cc": rng.choice([998, 1197, 1248, 1498], n).astype(float),
        "max_power_bhp": np.concatenate([rng.normal(85, 20, n - 1), [np.nan]]),
        "torque_nm": rng.normal(160, 40, n),
        "seats": rng.choice([5.0, 7.0], n),
        "km_per_year": rng.normal(12_000, 3_000, n),
        "fuel": rng.choice(["Petrol", "Diesel"], n),
        "seller_type": rng.choice(["Individual", "Dealer"], n),
        "transmission": rng.choice(["Manual", "Automatic"], n),
        "owner": rng.choice(["First Owner", "Second Owner"], n),
        "brand": rng.choice(["Maruti", "Hyundai", "Toyota"], n),
    })


def test_preprocessor_structure():
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)
    num_pipe = {t[0]: t[1] for t in pre.transformers}["num"]
    cat_pipe = {t[0]: t[1] for t in pre.transformers}["cat"]
    assert isinstance(num_pipe, Pipeline)
    assert isinstance(num_pipe.steps[0][1], SimpleImputer)
    assert isinstance(num_pipe.steps[1][1], StandardScaler)
    assert isinstance(cat_pipe.steps[0][1], SimpleImputer)
    ohe = cat_pipe.steps[1][1]
    assert isinstance(ohe, OneHotEncoder)
    assert ohe.handle_unknown == "ignore"


def test_numeric_pipeline_imputes_and_scales(sample_df):
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(sample_df)
    n_num = len(NUMERIC)
    numeric_part = X[:, :n_num]
    # imputed -> no NaN; scaled -> ~zero mean, ~unit std
    assert np.isfinite(numeric_part).all()
    assert np.allclose(numeric_part.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(numeric_part.std(axis=0), 1, atol=1e-8)


def test_onehot_expands_categories(sample_df):
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(sample_df)
    n_num = len(NUMERIC)
    cat_part = X[:, n_num:]
    # one-hot block is 0/1 and wider than the 5 source columns
    assert cat_part.min() == 0 and cat_part.max() == 1
    assert cat_part.shape[1] > len(CATEGORICAL)


def test_unknown_categories_do_not_crash(sample_df):
    pre = build_preprocessor().fit(sample_df)
    unseen = sample_df.copy()
    unseen["brand"] = "Totally New Brand"
    unseen["fuel"] = "Electric"
    X = pre.transform(unseen)          # must not raise
    assert np.isfinite(X).all()
    # unseen brand/fuel rows are all-zero in their one-hot blocks
    names = pre.get_feature_names_out()
    brand_idx = [i for i, name in enumerate(names) if "brand_" in name]
    assert len(brand_idx) > 0
    assert (X[0, brand_idx] == 0).all()
    fuel_idx = [i for i, name in enumerate(names) if "fuel_" in name]
    assert (X[0, fuel_idx] == 0).all()


def test_output_has_no_nan(sample_df):
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(sample_df)
    assert np.isfinite(X).all()


def test_fit_on_train_only_changes_stats(sample_df):
    """Imputation medians learned on the first half must not move when a
    differently-distributed second half is transformed (leakage guard)."""
    train, test = sample_df.iloc[:20], sample_df.iloc[20:]
    train = train.copy()
    train["mileage_kmpl"] = 30.0        # distinctive training median
    train.loc[train.index[:5], "mileage_kmpl"] = np.nan

    pre = build_preprocessor()
    pre.fit(train)
    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    median_before = imputer.statistics_[NUMERIC.index("mileage_kmpl")]

    pre.transform(test)                 # transforming test must NOT refit
    median_after = imputer.statistics_[NUMERIC.index("mileage_kmpl")]
    assert median_before == median_after == 30.0
