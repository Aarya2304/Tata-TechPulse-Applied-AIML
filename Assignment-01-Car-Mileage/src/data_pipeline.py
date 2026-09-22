"""Data loading, cleaning and leakage-free missing-value handling.

The raw UCI ``auto-mpg.data`` file contains 398 rows.  A handful of rows have
``horsepower`` recorded as ``"?"``; those are converted to NaN and imputed.

Leakage policy
--------------
Missing-value statistics are **learned from the training split only** and then
applied to both train and test data.  ``HorsepowerImputer`` is a scikit-learn
transformer, so it lives inside the model ``Pipeline`` and is re-fit on the
training folds of every cross-validation split automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------
def load_raw(path=None) -> pd.DataFrame:
    """Load the raw UCI Auto MPG file (whitespace separated, '?' = missing)."""
    path = config.RAW_DATA_PATH if path is None else path
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=config.COLUMN_NAMES,
        na_values=config.MISSING_MARKER,
        quotechar='"',
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: types, duplicates, impossible values, ordering.

    Rows whose *target* is missing are dropped (there are none in the raw
    file, but the guard keeps the pipeline robust).
    """
    df = df.copy()

    # Normalise the car name (kept only for reporting; not a model feature).
    df["car_name"] = df["car_name"].astype(str).str.strip().str.strip('"')

    # Enforce numeric dtypes (they should already be, but be defensive).
    for col in [config.TARGET] + config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop exact duplicate observations (same specs AND same name).
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df)

    # Drop rows with missing target (mpg) -- cannot be imputed safely.
    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)

    # Sanity-check value ranges.
    assert df[config.TARGET].between(0, 100).all(), "mpg out of plausible range"
    assert df["cylinders"].between(1, 16).all(), "cylinders out of plausible range"

    df.attrs["duplicates_removed"] = n_dupes
    return df


def load_and_clean(path=None) -> pd.DataFrame:
    """Convenience: raw load + clean in one call."""
    return clean(load_raw(path))


# ---------------------------------------------------------------------------
# Missing-value report (EDA helper)
# ---------------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column count of missing values (only non-zero columns)."""
    report = df.isna().sum()
    report = report[report > 0].sort_values(ascending=False)
    return report.to_frame(name="missing_count")


# ---------------------------------------------------------------------------
# Leakage-free imputation transformer
# ---------------------------------------------------------------------------
class HorsepowerImputer(BaseEstimator, TransformerMixin):
    """Impute missing ``horsepower`` using cylinder-group medians.

    Learned from the data passed to :meth:`fit` (the *training* split only):

    * median horsepower per ``cylinders`` group;
    * overall median horsepower as fallback for unseen cylinder counts.

    Other columns pass through untouched.
    """

    def __init__(self, group_col: str = config.IMPUTE_GROUP_COL):
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y=None):
        X = self._as_frame(X)
        self.overall_median_ = float(X["horsepower"].median())
        self.group_medians_ = (
            X.groupby(self.group_col)["horsepower"].median().to_dict()
        )
        # Remember which columns we saw at fit time (order stability).
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._as_frame(X).copy()
        missing_mask = X["horsepower"].isna()

        if missing_mask.any():
            # Fill with the median of the same cylinder group when available.
            mapped = X.loc[missing_mask, self.group_col].map(self.group_medians_)
            X.loc[missing_mask, "horsepower"] = mapped
            # Fallback for cylinder counts never seen during training.
            still_missing = X["horsepower"].isna()
            if still_missing.any():
                X.loc[still_missing, "horsepower"] = self.overall_median_

        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            input_features if input_features is not None else self.feature_names_in_,
            dtype=object,
        )

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
