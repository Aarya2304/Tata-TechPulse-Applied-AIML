"""Deterministic synthetic vehicle dataset for the MLOps workflow.

Generates a small, automotive-flavoured regression dataset: vehicle
attributes -> selling price.  The relationships are hand-designed and all
noise is drawn from a seeded NumPy generator, so the dataset is fully
reproducible and contains no missing values by construction.

IMPORTANT: this data is SYNTHETIC.  It exists purely to demonstrate the
MLOps lifecycle (training -> MLflow tracking -> packaging -> serving) and
must not be interpreted as a real market-price dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

_REQUIRED_COLUMNS = [*config.FEATURE_NAMES, config.TARGET_NAME]


def generate_dataset(
    n_samples: int = config.N_SAMPLES, seed: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Generate the synthetic vehicle dataset (deterministic given ``seed``).

    Designed relationships (all positive/negative signs match market
    intuition for used vehicles):

    * higher ``max_power_bhp``      -> higher price
    * higher ``engine_cc``          -> higher price
    * lower  ``vehicle_age``        -> higher price
    * lower  ``km_driven``          -> higher price
    * higher ``mileage_kmpl``       -> modestly higher price
    * multiplicative log-normal noise keeps prices positive and skewed,
      as real used-car prices are.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    rng = np.random.default_rng(seed)

    vehicle_age = rng.integers(0, 16, n_samples)  # 0-15 years old
    km_per_year = rng.normal(12_000, 4_000, n_samples).clip(2_000, 30_000)
    km_driven = np.round(
        vehicle_age * km_per_year + rng.normal(0, 3_000, n_samples)
    ).clip(500, 300_000)
    mileage_kmpl = rng.normal(18.0, 4.0, n_samples).clip(8.0, 35.0).round(1)
    engine_cc = rng.normal(1_500, 400, n_samples).clip(800, 3_000).round()
    # Power is realistically correlated with engine size (plus noise).
    max_power_bhp = (engine_cc / 15 + rng.normal(0, 15, n_samples)).clip(
        40, 400
    ).round(1)

    log_price = (
        np.log(650_000)
        + 0.55 * np.log(max_power_bhp / 100.0)
        + 0.35 * np.log(engine_cc / 1_500.0)
        - 0.075 * vehicle_age
        - 0.25 * np.log(km_driven / 50_000.0)
        + 0.15 * (mileage_kmpl - 18.0) / 5.0
        + rng.normal(0, 0.30, n_samples)
    )
    selling_price = np.exp(log_price).round(-2)  # nearest 100 INR

    return pd.DataFrame(
        {
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "mileage_kmpl": mileage_kmpl,
            "engine_cc": engine_cc,
            "max_power_bhp": max_power_bhp,
            config.TARGET_NAME: selling_price,
        }
    )


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the dataset is structurally invalid."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if df[list(_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("dataset contains missing values")
    if len(df) == 0:
        raise ValueError("dataset is empty")
    non_numeric = [
        c
        for c in _REQUIRED_COLUMNS
        if not np.issubdtype(df[c].dtype, np.number)
    ]
    if non_numeric:
        raise ValueError(f"non-numeric columns found: {non_numeric}")
    if not np.isfinite(df[list(_REQUIRED_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("dataset contains non-finite values")


def load_training_data(
    n_samples: int = config.N_SAMPLES, seed: int = config.RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix ``X`` and target ``y`` for training."""
    df = generate_dataset(n_samples=n_samples, seed=seed)
    validate_dataset(df)
    X = df[config.FEATURE_NAMES]
    y = df[config.TARGET_NAME]
    return X, y


def dataset_summary(df: pd.DataFrame) -> dict:
    """Small JSON-safe description of the dataset (for MLflow metadata)."""
    validate_dataset(df)
    return {
        "dataset_type": config.DATASET_TYPE,
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURE_NAMES),
        "feature_names": list(config.FEATURE_NAMES),
        "target_name": config.TARGET_NAME,
        "target_min": float(df[config.TARGET_NAME].min()),
        "target_max": float(df[config.TARGET_NAME].max()),
        "target_mean": float(df[config.TARGET_NAME].mean()),
        "missing_values": int(df[list(_REQUIRED_COLUMNS)].isna().sum().sum()),
    }
