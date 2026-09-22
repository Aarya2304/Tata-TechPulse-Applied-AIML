"""Central configuration: paths, constants and reproducibility settings."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "auto-mpg.data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_CSV_PATH = ARTIFACTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Dataset schema (UCI Auto MPG)
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]

TARGET = "mpg"
FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]
NUMERIC_FEATURES = FEATURES  # all model inputs are numeric (origin coded 1/2/3)

# UCI missing-value marker used in the raw file (e.g. in horsepower)
MISSING_MARKER = "?"

# Feature used to group-wise impute missing horsepower
IMPUTE_GROUP_COL = "cylinders"
