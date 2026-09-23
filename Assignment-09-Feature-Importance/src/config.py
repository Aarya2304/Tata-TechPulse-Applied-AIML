"""Central configuration for the feature-importance project.

All paths are relative; no absolute paths, no user-specific locations.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURE_IMPORTANCE_CSV = ARTIFACTS_DIR / "feature_importance.csv"
PERMUTATION_IMPORTANCE_CSV = ARTIFACTS_DIR / "permutation_importance.csv"
GROUPED_IMPORTANCE_CSV = ARTIFACTS_DIR / "grouped_feature_importance.csv"
MODEL_METRICS_JSON = ARTIFACTS_DIR / "model_metrics.json"
TOP_FEATURES_PNG = ARTIFACTS_DIR / "top_feature_importance.png"
TOP_PERMUTATION_PNG = ARTIFACTS_DIR / "top_permutation_importance.png"
COMPARISON_PNG = ARTIFACTS_DIR / "importance_comparison.png"
REPORT_MD = REPORTS_DIR / "feature_importance_report.md"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = 'CarDekho "Car details v3"'
# The raw CarDekho file already exists in this repository from Assignments 3
# and 4 (identical copies, same md5). It is reused STRICTLY read-only; this
# project copies nothing and modifies nothing there. When the local copies
# are unavailable (fresh clone), the dataset is downloaded from the public
# GitHub mirror instead.
DATASET_URL = (
    "https://raw.githubusercontent.com/imanishshahu/Car-Price-Prediction/"
    "main/Car%20details%20v3.csv"
)
SIBLING_DATASET_CANDIDATES = (
    PROJECT_ROOT.parent / "Assignment-04-Vehicle-Price-Prediction" / "data" / "car_data.csv",
    PROJECT_ROOT.parent / "Assignment-03-Data-Cleaning" / "data" / "car_data.csv",
)

TARGET = "selling_price"
# Original columns used as model inputs (identifiers such as the full vehicle
# name are excluded; the first token is engineered into `brand` instead).
RAW_NUMERIC_FEATURES = ["year", "km_driven", "seats"]
RAW_CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner", "brand"]
PARSED_NUMERIC_FEATURES = ["mileage_kmpl", "engine_cc", "max_power_bhp", "torque_nm"]
ENGINEERED_NUMERIC_FEATURES = ["vehicle_age", "km_per_year"]

TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
N_ESTIMATORS = 300
N_JOBS = -1

# ---------------------------------------------------------------------------
# Importance computation
# ---------------------------------------------------------------------------
PERMUTATION_SCORING = "neg_mean_absolute_error"  # tied to prediction error (INR)
PERMUTATION_N_REPEATS = 15
TOP_N = 20
N_REPETITIONS = 5  # seeds for the reproducibility check (42..46)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
PLOT_DPI = 120
TOP_N_BAR_COLORS = "#2c7fb8"
