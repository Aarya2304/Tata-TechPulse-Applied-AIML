"""Tests for the leakage-safe preprocessing pipelines."""

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


@pytest.fixture()
def sample_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "Air temperature [K]": rng.normal(300, 2, n),
        "Process temperature [K]": rng.normal(310, 1.5, n),
        "Rotational speed [rpm]": rng.normal(1540, 180, n),
        "Torque [Nm]": rng.normal(40, 10, n),
        "Tool wear [min]": rng.integers(0, 250, n).astype(float),
        "temp_diff_k": rng.normal(10, 1.5, n),
        "power_w": rng.normal(6300, 900, n),
        "torque_x_wear": rng.normal(4300, 2600, n),
        "Type": rng.choice(["L", "M", "H"], n),
    })


def test_preprocessor_structure():
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)
    pipes = {t[0]: t[1] for t in pre.transformers}
    assert isinstance(pipes["num"], Pipeline)
    assert isinstance(pipes["num"].steps[0][1], SimpleImputer)
    assert isinstance(pipes["num"].steps[1][1], StandardScaler)
    assert isinstance(pipes["cat"].steps[0][1], SimpleImputer)
    ohe = pipes["cat"].steps[1][1]
    assert isinstance(ohe, OneHotEncoder)
    assert ohe.handle_unknown == "ignore"


def test_transform_produces_finite_output(sample_df):
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(sample_df)
    assert np.isfinite(X).all()


def test_no_nan_after_transform_with_missing_values(sample_df):
    """Imputation must remove NaNs even when they appear at transform time."""
    dirty = sample_df.copy()
    dirty.loc[dirty.index[:5], "Torque [Nm]"] = np.nan
    dirty["Type"] = dirty["Type"].astype(object)
    dirty.loc[dirty.index[3], "Type"] = None
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(dirty)
    assert np.isfinite(X).all()


def test_numeric_output_is_scaled(sample_df):
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(sample_df)
    n_num = len(config.NUMERIC_FEATURES)
    num = X[:, :n_num]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(num.std(axis=0), 1, atol=1e-8)


def test_onehot_expands_type(sample_df):
    pre = build_preprocessor().fit(sample_df)
    X = pre.transform(sample_df)
    cat_block = X[:, len(config.NUMERIC_FEATURES):]
    assert cat_block.shape[1] == 3          # L / M / H
    assert set(np.unique(cat_block)) <= {0.0, 1.0}
    assert cat_block.sum(axis=1).min() == 1  # every row is exactly one type


def test_unknown_category_does_not_crash(sample_df):
    pre = build_preprocessor().fit(sample_df)
    unseen = sample_df.copy()
    unseen["Type"] = "Z"                    # never seen in training
    X = pre.transform(unseen)
    assert np.isfinite(X).all()
    names = pre.get_feature_names_out()
    type_idx = [i for i, n in enumerate(names) if n.startswith("cat__Type_")]
    assert (X[0, type_idx] == 0).all()      # all-zero block for unknown


def test_leakage_columns_not_in_transformer(sample_df):
    pre = build_preprocessor()
    used = [c for _, _, cols in pre.transformers if isinstance(cols, list)
            for c in cols]
    assert config.TARGET not in used
    for col in config.LEAKAGE_COLUMNS + config.ID_COLUMNS:
        assert col not in used


def test_stats_frozen_after_fit(sample_df):
    """Refit-free transforms must not move learned statistics (leakage)."""
    train = sample_df.iloc[:20].copy()
    train["Torque [Nm]"] = 55.0             # distinctive training median
    pre = build_preprocessor().fit(train)
    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    median_idx = config.NUMERIC_FEATURES.index("Torque [Nm]")
    before = imputer.statistics_[median_idx]
    pre.transform(sample_df.iloc[20:])      # test data must not refit
    assert imputer.statistics_[median_idx] == before == 55.0
