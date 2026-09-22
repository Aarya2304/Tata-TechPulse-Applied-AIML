"""Data cleaning: type parsing, domain checks, outlier detection & capping.

Design decisions (all documented in the README):
* Numeric strings ("23.4 kmpl", "1248 CC", "74 bhp") are parsed to floats.
* Torque strings come in three formats; kgm values are converted to Nm.
* Exact duplicate listings are dropped (same specs = same physical car).
* Domain sanity checks treat impossible values (used car with 1 km, 0 kmpl
  fuel economy, 0 bhp power) as missing data, NOT as outliers to delete.
* Outliers are detected with the 1.5*IQR rule; only extreme magnitudes are
  winsorized to the IQR fence, and thresholds are learned from training
  data via the ``IQRCapper`` transformer (leakage-free).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

_TORQUE_KGM = "kgm"
_NM_PER_KGM = 9.80665


# ---------------------------------------------------------------------------
# Type parsing / feature extraction
# ---------------------------------------------------------------------------
def parse_numeric_value(value) -> float:
    """Extract the leading float from strings like '23.4 kmpl' or '1248 CC'."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else np.nan


def parse_torque(value) -> float:
    """Parse a torque string to Newton-metres.

    Handles the three formats present in the dataset:
      '190Nm@ 2000rpm'            -> 190.0 Nm
      '22.4 kgm at 1750-2750rpm'  -> 22.4 * 9.80665 = 219.7 Nm
      '12.7@ 2,700(kgm@ rpm)'     -> 12.7 * 9.80665 = 124.5 Nm
    """
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    value_num = float(match.group())
    if _TORQUE_KGM in text or "(kgm" in text:
        value_num *= _NM_PER_KGM
    return value_num


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with parsed numeric columns and derived ``brand``.

    The raw string columns (``name``, ``mileage``, ``engine``, ``max_power``,
    ``torque``) are consumed by the parsing and dropped, leaving a frame
    whose columns are all either typed features or the target.
    """
    out = df.copy()
    for src_col, (dst_col, _unit) in config.PARSED_NUM_COLUMNS.items():
        if src_col == "torque":
            out[dst_col] = out[src_col].map(parse_torque)
        else:
            out[dst_col] = out[src_col].map(parse_numeric_value)

    out["brand"] = out["name"].astype(str).str.strip().str.split().str[0]
    out = out.drop(columns=["name", "mileage", "engine", "max_power",
                            "torque"])
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> tuple:
    """Drop exact duplicate rows; returns (deduped_df, n_removed)."""
    n_before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, n_before - len(deduped)


# ---------------------------------------------------------------------------
# Domain sanity checks
# ---------------------------------------------------------------------------
def apply_domain_checks(df: pd.DataFrame) -> tuple:
    """Flag impossible automotive values as MISSING (not outliers).

    Rules (justified in README):
    * ``km_driven < MIN_PLAUSIBLE_KM_DRIVEN`` for a used car is a data-entry
      error -> NaN (imputed later).
    * ``mileage_kmpl == 0`` and ``max_power_bhp == 0`` are physically
      impossible for a listed car -> NaN.
    """
    out = df.copy()
    n_km = int((out["km_driven"] < config.MIN_PLAUSIBLE_KM_DRIVEN).sum())
    out.loc[out["km_driven"] < config.MIN_PLAUSIBLE_KM_DRIVEN,
            "km_driven"] = np.nan

    n_zero = {}
    for col in config.ZERO_AS_MISSING_COLUMNS:
        n_zero[col] = int((out[col] == 0).sum())
        out.loc[out[col] == 0, col] = np.nan
    return out, {"km_driven_below_min": n_km, **n_zero}


# ---------------------------------------------------------------------------
# Missing-value handling (report-friendly, dataframe level)
# ---------------------------------------------------------------------------
def missing_before_after(df_before: pd.DataFrame,
                         df_after: pd.DataFrame) -> pd.DataFrame:
    """Build the before/after missing-values report table."""
    rows = []
    cols = df_before.columns.union(df_after.columns)
    for col in cols:
        b = int(df_before[col].isna().sum()) if col in df_before else 0
        a = int(df_after[col].isna().sum()) if col in df_after else 0
        rows.append({
            "column": col,
            "missing_before": b,
            "missing_pct_before": round(100 * b / max(len(df_before), 1), 3),
            "missing_after": a,
            "missing_pct_after": round(100 * a / max(len(df_after), 1), 3),
        })
    return pd.DataFrame(rows)


def impute_dataframe(df: pd.DataFrame,
                     numeric_cols: list,
                     categorical_cols: list) -> pd.DataFrame:
    """Simple dataframe-level imputation used for the cleaning-stage report.

    (The ML pipeline in ``preprocessing.py`` re-implements this leakage-free
    with sklearn imputers; this helper is for the standalone cleaned CSV.)
    """
    out = df.copy()
    for col in numeric_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    for col in categorical_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out


# ---------------------------------------------------------------------------
# Outlier detection (IQR)
# ---------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float = config.IQR_K) -> tuple:
    """Return (q1, q3, iqr, lower_fence, upper_fence) for a numeric series."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1, q3, iqr, q1 - k * iqr, q3 + k * iqr


def detect_outliers_iqr(df: pd.DataFrame,
                        columns: list,
                        k: float = config.IQR_K) -> tuple:
    """Count outliers per column; returns (report_df, bounds_dict)."""
    rows, bounds = [], {}
    for col in columns:
        s = df[col].dropna()
        q1, q3, iqr, lo, hi = iqr_bounds(s, k)
        n_out = int(((s < lo) | (s > hi)).sum())
        bounds[col] = {"q1": q1, "q3": q3, "iqr": iqr,
                       "lower": lo, "upper": hi}
        rows.append({
            "feature": col,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_fence": round(lo, 3),
            "upper_fence": round(hi, 3),
            "outlier_count": n_out,
            "outlier_pct": round(100 * n_out / max(len(s), 1), 3),
        })
    return pd.DataFrame(rows), bounds


# ---------------------------------------------------------------------------
# Leakage-free IQR capper transformer
# ---------------------------------------------------------------------------
class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns to [Q1 - k*IQR, Q3 + k*IQR].

    Bounds are learned in ``fit`` (call on TRAINING data only) and reused in
    ``transform``; placed inside a sklearn Pipeline/ColumnTransformer this is
    structurally leakage-free.
    """

    def __init__(self, k: float = config.IQR_K):
        self.k = k

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = [str(c) for c in X.columns]
        # Bounds are stored POSITIONALLY: within a sklearn Pipeline the
        # column order seen by fit and transform is guaranteed identical,
        # whether the input is a named DataFrame or a numpy array.
        self.lower_ = []
        self.upper_ = []
        for i in range(X.shape[1]):
            s = X.iloc[:, i].dropna()
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            self.lower_.append(q1 - self.k * iqr)
            self.upper_.append(q3 + self.k * iqr)
        return self

    def transform(self, X):
        X = self._as_frame(X)
        values = X.to_numpy(dtype=float, copy=True)
        for i in range(values.shape[1]):
            values[:, i] = np.clip(values[:, i], self.lower_[i],
                                   self.upper_[i])
        return pd.DataFrame(values, index=X.index, columns=X.columns)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            input_features if input_features is not None
            else self.feature_names_in_, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def count_capped_values(df_original: pd.DataFrame,
                        df_capped: pd.DataFrame) -> dict:
    """Count values changed by winsorization per column."""
    changed = {}
    for col in df_original.columns:
        a = df_original[col]
        b = df_capped[col]
        changed[col] = int(((a != b) & a.notna() & b.notna()).sum())
    return changed
