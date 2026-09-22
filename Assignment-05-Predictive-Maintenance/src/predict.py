"""Prediction CLI: load the saved pipeline and classify sample machines.

Usage:
    python -m src.predict                 # built-in demo sensor rows
    python -m src.predict --csv file.csv  # batch mode from a CSV

The CSV must contain the raw sensor columns
(Type, Air temperature [K], Process temperature [K],
 Rotational speed [rpm], Torque [Nm], Tool wear [min]);
engineered features are recomputed here with the same functions as training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.feature_engineering import add_engineered_features

# Realistic AI4I-style operating points (value ranges from the dataset).
DEMO_MACHINES = [
    # nominal low-wear operation -> expected: no failure
    {"Type": "L", "Air temperature [K]": 298.9,
     "Process temperature [K]": 309.0, "Rotational speed [rpm]": 1500,
     "Torque [Nm]": 38.5, "Tool wear [min]": 15},
    # high torque through a heavily worn tool -> expected: failure risk
    {"Type": "L", "Air temperature [K]": 298.9,
     "Process temperature [K]": 309.3, "Rotational speed [rpm]": 1300,
     "Torque [Nm]": 65.0, "Tool wear [min]": 220},
    # low temperature differential but moderate power -> outside the
    # heat-dissipation danger zone, model should say no failure
    {"Type": "M", "Air temperature [K]": 302.5,
     "Process temperature [K]": 304.0, "Rotational speed [rpm]": 2800,
     "Torque [Nm]": 15.0, "Tool wear [min]": 40},
]


def predict_frame(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for raw sensor rows and predict failure probs."""
    feats = add_engineered_features(raw_df)
    feats = feats[config.FEATURES]
    prob = model.predict_proba(feats)[:, 1]
    pred = (prob >= config.DECISION_THRESHOLD).astype(int)
    return pd.DataFrame({
        "failure_probability": prob,
        "predicted_failure": pred,
        "prediction_label": ["FAILURE" if p else "NO FAILURE" for p in pred],
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict machine failure with the saved model.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV of raw sensor rows to score.")
    args = parser.parse_args()

    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {config.BEST_MODEL_PATH}.\n"
            "Run 'python -m src.train' first.")

    model = joblib.load(config.BEST_MODEL_PATH)
    print(f"Loaded model from {config.BEST_MODEL_PATH}")

    if args.csv:
        path = Path(args.csv)
        if not path.exists():
            raise SystemExit(f"CSV file not found: {path}")
        raw = pd.read_csv(path)
        preds = predict_frame(model, raw)
        preds.insert(0, "row", range(len(preds)))
        out = path.with_name(path.stem + "_predictions.csv")
        preds.to_csv(out, index=False)
        print(f"\nWrote {len(preds)} predictions to {out}")
        print(preds.to_string(index=False))
        return

    demo = pd.DataFrame(DEMO_MACHINES)
    preds = predict_frame(model, demo)
    print("\nDemo sensor rows (actual saved model, no hardcoded values):\n")
    for i, (row, p) in enumerate(zip(DEMO_MACHINES,
                                     preds.itertuples(index=False))):
        print(f"  Machine {i + 1}: Type={row['Type']} "
              f"speed={row['Rotational speed [rpm]']} rpm, "
              f"torque={row['Torque [Nm]']} Nm, "
              f"wear={row['Tool wear [min]']} min")
        print(f"      -> {p.prediction_label}  "
              f"(failure probability: {p.failure_probability:.3f})")


if __name__ == "__main__":
    main()
