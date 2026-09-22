"""Tests for engineered features and leak-safe feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import add_engineered_features, build_feature_frame


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["L0001", "M0002"],
        "Type": ["L", "M"],
        "Air temperature [K]": [298.9, 302.5],
        "Process temperature [K]": [309.0, 304.0],
        "Rotational speed [rpm]": [1500, 2800],
        "Torque [Nm]": [38.5, 15.0],
        "Tool wear [min]": [15, 40],
        "Machine failure": [0, 1],
        "TWF": [0, 0], "HDF": [0, 1], "PWF": [0, 0],
        "OSF": [0, 0], "RNF": [0, 0],
    })


def test_engineered_columns_present(sample_df):
    out = add_engineered_features(sample_df)
    for col in config.ENGINEERED_FEATURES:
        assert col in out.columns


def test_temp_diff_is_process_minus_air(sample_df):
    out = add_engineered_features(sample_df)
    expected = sample_df["Process temperature [K]"] - \
        sample_df["Air temperature [K]"]
    assert np.allclose(out["temp_diff_k"], expected)
    assert out.loc[0, "temp_diff_k"] == pytest.approx(10.1)


def test_power_matches_physics(sample_df):
    out = add_engineered_features(sample_df)
    expected = (sample_df["Torque [Nm]"] * 2 * np.pi
                * sample_df["Rotational speed [rpm]"] / 60.0)
    assert np.allclose(out["power_w"], expected)
    # 38.5 Nm at 1500 rpm -> 2*pi*1500/60 * 38.5 = 6047.6... W
    assert out.loc[0, "power_w"] == pytest.approx(6047.57, rel=1e-4)


def test_torque_x_wear(sample_df):
    out = add_engineered_features(sample_df)
    assert out.loc[0, "torque_x_wear"] == pytest.approx(38.5 * 15)
    assert out.loc[1, "torque_x_wear"] == pytest.approx(15.0 * 40)


def test_engineering_is_rowwise_target_independent(sample_df):
    """Shuffling the target must not change any engineered feature."""
    out1 = add_engineered_features(sample_df)
    flipped = sample_df.copy()
    flipped["Machine failure"] = 1 - flipped["Machine failure"]
    out2 = add_engineered_features(flipped)
    for col in config.ENGINEERED_FEATURES:
        assert np.allclose(out1[col], out2[col])


def test_feature_frame_excludes_leakage_and_ids(sample_df):
    X, y = build_feature_frame(sample_df)
    for col in config.LEAKAGE_COLUMNS + config.ID_COLUMNS:
        assert col not in X.columns
    assert config.TARGET not in X.columns
    assert list(X.columns) == config.FEATURES
    assert list(y) == [0, 1]


def test_feature_frame_on_real_data():
    from src.data_loader import load_raw
    df = load_raw()
    X, y = build_feature_frame(df)
    assert len(X) == len(df) == len(y)
    assert X.notna().all().all()
    assert set(y.unique()) == {0, 1}
    assert config.TARGET not in X.columns
    for col in config.LEAKAGE_COLUMNS:
        assert col not in X.columns
