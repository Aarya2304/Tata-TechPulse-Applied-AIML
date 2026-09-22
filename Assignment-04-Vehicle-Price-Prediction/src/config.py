"""Central configuration for the vehicle price prediction project."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "car_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_vehicle_price_model.joblib"
MODEL_META_PATH = MODELS_DIR / "best_model_metadata.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_CSV_PATH = ARTIFACTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Reference year used to derive vehicle_age from `year`.
# 2020 = the newest model year present in the dataset, so the engineered
# age is non-negative for every row and reproducible across runs.
REFERENCE_YEAR = 2020

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "name", "year", "selling_price", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine", "max_power", "torque",
    "seats",
]
TARGET = "selling_price"

NUMERIC_FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
    "km_per_year",
]
CATEGORICAL_FEATURES = [
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Target transformation
# ---------------------------------------------------------------------------
# "log1p" fits the model on log1p(selling_price) and metrics are reported
# after expm1 inverse-transform, i.e. in original rupee units.
TARGET_TRANSFORM = "log1p"

# ---------------------------------------------------------------------------
# Prediction CLI defaults
# ---------------------------------------------------------------------------
EXAMPLE_CAR = {
    "year": 2017,
    "km_driven": 45000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "23.4 kmpl",
    "engine": "1248 CC",
    "max_power": "74 bhp",
    "torque": "190Nm@ 2000rpm",
    "seats": 5,
    "name": "Maruti Swift Dzire VDI",
}
