"""Pytest scaffolding: make ``src`` importable from any working directory.

The tests are hermetic: MLflow tests use an isolated temporary tracking
store, and no test talks to Docker or a live server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before mlflow is imported anywhere in the test process.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
