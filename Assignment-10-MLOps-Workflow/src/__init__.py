"""TechPulse Assignment 10 - MLOps workflow simulation package.

Sets the MLflow file-store opt-in *before* any submodule (and therefore
mlflow) is imported: mlflow >= 3 places the local filesystem tracking
backend in maintenance mode and raises unless this environment flag is
present.  The assignment explicitly uses the local ``mlruns/`` store so
that ``mlflow ui --backend-store-uri ./mlruns`` works without a server.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from src import config  # noqa: E402,F401  (re-exported for convenience)
