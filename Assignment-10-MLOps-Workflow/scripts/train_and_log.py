"""Standalone entry point: full training + MLflow logging.

Equivalent to ``python -m src.train`` but packaged as a script so it can
be invoked from CI, the Makefile or ``scripts/deploy.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as ``python scripts/train_and_log.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train_and_log  # noqa: E402


def main() -> None:
    result = train_and_log()
    print(
        f"MLflow run {result['run_id']} | "
        f"MAE={result['metrics']['mae']:.2f} "
        f"RMSE={result['metrics']['rmse']:.2f} "
        f"R2={result['metrics']['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
