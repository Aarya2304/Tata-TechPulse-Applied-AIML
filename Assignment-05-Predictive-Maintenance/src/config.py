"""Central configuration for the predictive maintenance project."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_predictive_maintenance_model.joblib"
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_CSV_PATH = ARTIFACTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Dataset schema (AI4I 2020, UCI id 601)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]",
    "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
    "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]
TARGET = "Machine failure"

# Identifier columns: row number and per-row serial number -> no predictive
# value; "Product ID" also embeds the quality variant already kept in `Type`.
ID_COLUMNS = ["UDI", "Product ID"]

# Failure-mode indicator columns (TWF/HDF/PWF/OSF/RNF). These are the
# *components* of the target: they are only known once a failure has been
# diagnosed, so using them as inputs would be target leakage.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Raw sensor/operational inputs retained for modelling.
SENSOR_COLUMNS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

# ---------------------------------------------------------------------------
# Engineered features (see feature_engineering.py for rationale)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES = [
    "temp_diff_k",       # process - air temperature  (heat dissipation)
    "power_w",           # mechanical power from torque x speed
    "torque_x_wear",     # torque x tool wear         (tool overload)
]

NUMERIC_FEATURES = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]",
] + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Model selection criterion (fixed BEFORE any model was trained)
# ---------------------------------------------------------------------------
# Primary: failure-class F1 on the held-out test set (threshold 0.5).
# Tie-breakers, in order: PR-AUC (average precision), then failure recall.
SELECTION_METRIC = "f1"
SELECTION_TIEBREAKERS = ["average_precision", "recall"]
DECISION_THRESHOLD = 0.5
