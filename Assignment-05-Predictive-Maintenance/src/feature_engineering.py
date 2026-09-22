"""Feature engineering for predictive maintenance.

Adds a small set of physically motivated features. Everything here is
row-wise and target-independent: no engineered feature reads
``Machine failure`` or any failure-mode column, so nothing can leak.

Engineered features
-------------------
* ``temp_diff_k``      = Process temperature - Air temperature.
  The machine must maintain a temperature differential; a collapsing
  differential signals cooling problems (the dataset's heat-dissipation
  failure mode is triggered when this difference falls too low).
* ``power_w``          = Torque [Nm] x 2*pi x Rotational speed [rpm] / 60.
  Mechanical power in watts. Over-power operation stresses the spindle
  (power-failure mode is triggered by power outside the normal range).
* ``torque_x_wear``    = Torque [Nm] x Tool wear [min].
  Cutting force applied through a worn tool; tool-overload wear failure
  is driven by exactly this combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the documented engineered features; returns a new frame."""
    out = df.copy()
    out["temp_diff_k"] = (out["Process temperature [K]"]
                          - out["Air temperature [K]"])
    out["power_w"] = (out["Torque [Nm]"] * 2.0 * np.pi
                      * out["Rotational speed [rpm]"] / 60.0)
    out["torque_x_wear"] = out["Torque [Nm]"] * out["Tool wear [min]"]
    return out


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model inputs and target.

    Explicitly excludes UDI / Product ID (identifiers) and the five
    failure-mode indicator columns (TWF/HDF/PWF/OSF/RNF) because they are
    components of the target and only knowable after diagnosis.
    """
    feats = add_engineered_features(df)
    X = feats[config.FEATURES]
    y = feats[config.TARGET].astype(int)
    return X, y
