"""Tests for both feature-importance methods and their artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_and_prepare, make_splits
from src.importance import (
    compare_methods,
    group_to_original_features,
    impurity_importance,
    permutation_importance_df,
    transformed_feature_names,
)
from src.model import build_pipeline


@pytest.fixture(scope="module")
def computed():
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = make_splits(X, y)
    pipeline = build_pipeline().fit(X_train, y_train)
    imp = impurity_importance(pipeline)
    perm = permutation_importance_df(pipeline, X_test, y_test, n_repeats=3)
    grouped = group_to_original_features(imp, perm)
    comp = compare_methods(imp, perm)
    return pipeline, X_test, y_test, imp, perm, grouped, comp


def test_transformed_names_cover_all_features(computed):
    pipeline, *_ = computed
    names = transformed_feature_names(pipeline)
    # one-hot expands the categoricals, so there are more columns than inputs
    assert len(names) > 12
    assert "vehicle_age" in names and "brand_Maruti" in names


def test_impurity_importance_finite_nonnegative_sorted(computed):
    _, _, _, imp, _, _, _ = computed
    assert len(imp) > 0
    assert np.isfinite(imp["importance"]).all()
    assert (imp["importance"] >= 0).all()
    assert imp["importance"].is_monotonic_decreasing
    assert imp["importance"].sum() == pytest.approx(1.0, abs=1e-6)  # sklearn normalises


def test_permutation_importance_finite_sorted(computed):
    _, _, _, _, perm, _, _ = computed
    assert len(perm) > 0
    assert np.isfinite(perm["importance_mean"]).all()
    assert (perm["importance_std"] >= 0).all()
    assert perm["importance_mean"].is_monotonic_decreasing


def test_grouped_importance_sums_to_one(computed):
    _, _, _, imp, perm, grouped, _ = computed
    # grouping one-hot columns back must preserve total impurity mass
    assert grouped["impurity_importance"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (grouped["n_transformed_columns"] >= 1).all()
    assert "vehicle_age" in set(grouped["original_feature"])


def test_comparison_has_both_methods(computed):
    _, _, _, _, _, _, comp = computed
    assert len(comp) > 0
    assert comp["impurity_norm"].between(0, 1).all()
    assert comp["permutation_norm"].between(0, 1).all()


def test_permutation_reproducible_same_seed(computed):
    _, X_test, y_test, _, perm, _, _ = computed
    # recompute with a freshly fitted pipeline + same seed: identical table
    X, y, _ = load_and_prepare()
    X_train, _, y_train, _ = make_splits(X, y)
    pipeline = build_pipeline().fit(X_train, y_train)
    perm2 = permutation_importance_df(pipeline, X_test, y_test, n_repeats=3, random_state=42)
    pd.testing.assert_series_equal(
        perm["importance_mean"].round(10), perm2["importance_mean"].round(10)
    )


def test_expected_artifacts_exist():
    expected = [
        config.FEATURE_IMPORTANCE_CSV,
        config.PERMUTATION_IMPORTANCE_CSV,
        config.GROUPED_IMPORTANCE_CSV,
        config.MODEL_METRICS_JSON,
        config.TOP_FEATURES_PNG,
        config.TOP_PERMUTATION_PNG,
        config.COMPARISON_PNG,
    ]
    missing = [p for p in expected if not Path(p).exists()]
    if missing:
        pytest.skip(f"artifacts not generated yet (run python -m src.main): {missing}")
    for p in expected:
        assert Path(p).stat().st_size > 0


def test_saved_top_tables_are_sorted_correctly():
    if not config.FEATURE_IMPORTANCE_CSV.exists():
        pytest.skip("run python -m src.main first")
    imp = pd.read_csv(config.FEATURE_IMPORTANCE_CSV)
    perm = pd.read_csv(config.PERMUTATION_IMPORTANCE_CSV)
    assert imp["importance"].is_monotonic_decreasing
    assert perm["importance_mean"].is_monotonic_decreasing
    assert set(imp["method"]) == {"impurity"}
    assert set(perm["method"]) == {"permutation"}
    grouped = pd.read_csv(config.GROUPED_IMPORTANCE_CSV)
    assert grouped["impurity_importance"].sum() == pytest.approx(1.0, abs=1e-4)
