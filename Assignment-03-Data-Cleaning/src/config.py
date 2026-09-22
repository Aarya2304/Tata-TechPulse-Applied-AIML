"""Central configuration: paths, column specs, domain rules, pipeline knobs."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "car_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PROCESSED_DATA_PATH = ARTIFACTS_DIR / "processed_data.csv"
DATA_PROFILE_PATH = REPORTS_DIR / "data_profile.json"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.json"
MISSING_REPORT_CSV_PATH = REPORTS_DIR / "missing_values_before_after.csv"
OUTLIER_REPORT_CSV_PATH = REPORTS_DIR / "outliers_report.csv"
LEAKAGE_REPORT_PATH = REPORTS_DIR / "leakage_verification.json"

# ---------------------------------------------------------------------------
# Dataset schema (CarDekho "Car details v3")
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "name", "year", "selling_price", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine", "max_power", "torque",
    "seats",
]

TARGET = "selling_price"          # kept for context; NOT modeled here

# Parsed numeric features derived from string columns
PARSED_NUM_COLUMNS = {
    "mileage": ("mileage_kmpl", "kmpl / km/kg numeric value"),
    "engine": ("engine_cc", "CC"),
    "max_power": ("max_power_bhp", "bhp"),
    "torque": ("torque_nm", "Nm (kgm converted at 9.80665)"),
}

# Final engineered numeric feature set (before scaling)
NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner", "brand"]

# Columns dropped from the feature set (identifier / raw duplicate info)
ID_COLUMNS = ["name"]

# ---------------------------------------------------------------------------
# Cleaning / domain rules (documented in README)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Used cars with fewer than this many km are considered data-entry anomalies
# (a listed used car with 1 km is not plausible); the value is treated as
# missing and imputed rather than deleted.
MIN_PLAUSIBLE_KM_DRIVEN = 100.0

# A used car cannot have zero fuel economy or zero power -> treat as missing.
ZERO_AS_MISSING_COLUMNS = ["mileage_kmpl", "max_power_bhp"]

# IQR multiplier for outlier detection/capping
IQR_K = 1.5

# Features subject to IQR winsorization (extreme magnitudes, right-skewed).
# year / seats / engine_cc / torque are retained uncapped (see README).
WINSORIZE_FEATURES = ["km_driven", "selling_price", "mileage_kmpl",
                      "max_power_bhp"]

# ---------------------------------------------------------------------------
# Scaling comparison
# ---------------------------------------------------------------------------
SCALER_COMPARISON_FEATURES = ["km_driven", "selling_price", "mileage_kmpl",
                              "max_power_bhp"]
