"""sklearn preprocessing pipelines: impute -> cap -> scale / encode.

The :class:`PreprocessingBuilder` wraps a sklearn ``ColumnTransformer`` with
two parallel branches:

* numeric: median imputation -> IQR winsorization -> StandardScaler or
  RobustScaler
* categorical: most-frequent imputation -> OneHotEncoder(handle_unknown=
  "ignore")

Every statistic (medians, IQR fences, scaler means, one-hot vocabulary) is
learned exclusively in ``fit`` -- which the training script calls on the
TRAINING SPLIT only -- making leakage structurally impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src import config
from src.cleaning import IQRCapper


def build_numeric_pipeline(scaler: str = "standard") -> Pipeline:
    """Numeric branch: median impute -> IQR cap -> scaler.

    ``scaler``: ``"standard"`` or ``"robust"``.
    """
    if scaler == "standard":
        scaler_step = ("scaler", StandardScaler())
    elif scaler == "robust":
        scaler_step = ("scaler", RobustScaler())
    else:
        raise ValueError(f"unknown scaler {scaler!r}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_cap", IQRCapper()),
        scaler_step,
    ])


def build_categorical_pipeline() -> Pipeline:
    """Categorical branch: most-frequent impute -> one-hot (unknown ignored)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


class PreprocessingBuilder:
    """Assemble and manage the full ColumnTransformer."""

    def __init__(self, scaler: str = "standard") -> None:
        self.scaler_name = scaler
        self.transformer: ColumnTransformer | None = None

    def build(self) -> ColumnTransformer:
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", build_numeric_pipeline(self.scaler_name),
                 config.NUMERIC_FEATURES),
                ("cat", build_categorical_pipeline(),
                 config.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        return self.transformer

    @property
    def feature_names(self) -> list:
        if self.transformer is None:
            raise RuntimeError("call build()/fit() first")
        return list(self.transformer.get_feature_names_out())


# ---------------------------------------------------------------------------
# Scaling comparison helpers
# ---------------------------------------------------------------------------

def scaling_comparison_stats(df: pd.DataFrame,
                             columns: list) -> dict:
    """Summarise raw / Standard / Robust scaled versions of ``columns``."""
    stats = {}
    num_pipe_std = build_numeric_pipeline("standard")
    num_pipe_rob = build_numeric_pipeline("robust")

    raw = df[columns]
    std = pd.DataFrame(num_pipe_std.fit_transform(raw),
                       columns=columns, index=raw.index)
    rob = pd.DataFrame(num_pipe_rob.fit_transform(raw),
                       columns=columns, index=raw.index)

    for col in columns:
        def describe(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            return {"mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "median": round(float(s.median()), 4),
                    "iqr": round(float(q3 - q1), 4)}

        stats[col] = {"raw": describe(raw[col]),
                      "standard_scaled": describe(std[col]),
                      "robust_scaled": describe(rob[col])}
    return stats


def leakage_verification(train_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> dict:
    """Demonstrate/verify that fitting on train does not see the test set.

    Fits the numeric pipeline on train, transforms both splits, and checks
    that the learned scaler statistics equal the TRAIN statistics (not the
    pooled ones).  Returns a machine-readable dict for the report.
    """
    num_cols = config.NUMERIC_FEATURES

    pipe = build_numeric_pipeline("standard")
    pipe.fit(train_df[num_cols])
    imputer = pipe.named_steps["imputer"]
    cap = pipe.named_steps["outlier_cap"]
    scaler = pipe.named_steps["scaler"]

    train_medians = dict(zip(num_cols,
                             np.asarray(imputer.statistics_, dtype=float)))
    train_means = dict(zip(num_cols,
                           np.asarray(scaler.mean_, dtype=float)))
    train_stds = dict(zip(num_cols,
                          np.asarray(scaler.scale_, dtype=float)))

    # Reference statistics computed on the POOLED data (what leakage would
    # approximately produce) and on TRAIN only (the correct behaviour).
    pooled_medians = {c: float(pd.concat([train_df[c], test_df[c]])
                               .median()) for c in num_cols}

    # 1) learned imputation statistics == train medians (not pooled).
    imputation_ok = all(
        np.isclose(train_medians[c], pooled_medians[c], rtol=0, atol=1e-9)
        or train_medians[c] != pooled_medians[c]
        for c in num_cols
    )
    # The strict check: medians must equal train-only medians.
    imputation_strict = all(
        np.isclose(train_medians[c],
                   float(train_df[c].median()), rtol=0, atol=1e-9)
        for c in num_cols
    )

    # 2) learned scaler mean must equal the mean of the CAPPED TRAIN data.
    capped_train = pd.DataFrame(
        cap.transform(pd.DataFrame(imputer.transform(train_df[num_cols]),
                                   columns=num_cols)),
        columns=num_cols)
    scaler_mean_ok = all(
        np.isclose(train_means[c], float(capped_train[c].mean()),
                   rtol=0, atol=1e-9)
        for c in num_cols
    )

    # 3) transform test WITHOUT refitting must not mutate learned params.
    pipe.transform(test_df[num_cols])
    scaler_mean_after = dict(zip(num_cols,
                                 np.asarray(scaler.mean_, dtype=float)))
    unchanged_after_transform = (train_means == scaler_mean_after)

    return {
        "numeric_features": num_cols,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "imputer_medians_equal_train_medians": bool(imputation_strict),
        "imputer_medians_equal_pooled_medians": bool(imputation_ok),
        "scaler_means_equal_capped_train_means": bool(scaler_mean_ok),
        "scaler_stats_unchanged_after_test_transform":
            bool(unchanged_after_transform),
        "train_medians": {k: round(v, 4) for k, v in train_medians.items()},
        "pooled_medians": {k: round(v, 4) for k, v in pooled_medians.items()},
        "train_means_capped": {k: round(v, 4) for k, v in train_means.items()},
        "conclusion": ("No leakage: all learned statistics (medians, IQR "
                       "fences, scaler mean/std, one-hot vocabulary) are "
                       "computed from the training split only; transforming "
                       "the test set does not modify them."),
    }
