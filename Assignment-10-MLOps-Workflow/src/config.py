"""Central configuration for the MLOps workflow project.

Single source of truth for the application version, reproducibility
seeds, ML/model hyperparameters, dataset characteristics, MLflow
settings and every filesystem path.  All paths are derived from this
file's location, so the project contains no absolute or machine-specific
paths and works from any checkout location.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Application version (one central location, exposed via /version)
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Synthetic dataset configuration
# ---------------------------------------------------------------------------
DATASET_TYPE = "synthetic"
N_SAMPLES = 4000
FEATURE_NAMES = [
    "vehicle_age",
    "km_driven",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
]
TARGET_NAME = "selling_price"

# Lightweight settings used by CI, tests and the container bootstrap.
SMOKE_N_SAMPLES = 600
SMOKE_N_ESTIMATORS = 30

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_TYPE = "RandomForestRegressor"
N_ESTIMATORS = 100

# ---------------------------------------------------------------------------
# MLflow (local, file-based tracking -- no remote server required)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "TechPulse-Assignment-10"
RUN_NAME = "random-forest-vehicle-price"

# ---------------------------------------------------------------------------
# Paths (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODEL_PATH = ARTIFACTS_DIR / "vehicle_price_rf.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
FEATURE_INFO_PATH = ARTIFACTS_DIR / "feature_info.json"
EVALUATION_PATH = ARTIFACTS_DIR / "evaluation.txt"

# Local file-based MLflow tracking store (compatible with:
#   mlflow ui --backend-store-uri ./mlruns
# run from the project root).
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()

# ---------------------------------------------------------------------------
# API / deployment
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
DOCKER_IMAGE = f"techpulse-mlops:{APP_VERSION}"
DOCKER_CONTAINER = "techpulse-ml-api"
