"""Prediction script: load the saved best model and predict car mileage (mpg).

Usage
-----
Run from the project root (``Assignment-01-Car-Mileage/``):

    # interactive demo with three built-in example cars
    python predict.py

    # predict from key=value specs
    python predict.py --specs cylinders=4 displacement=120 horsepower=88 \
        weight=2130 acceleration=14.5 model_year=82 origin=3

    # predict from a CSV file (same columns as features, optional car_name)
    python predict.py --csv path/to/cars.csv
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from src import config

# Three example cars (first, median and one of the most efficient in the set).
EXAMPLE_CARS = [
    {"car_name": "chevrolet chevelle malibu (1970)",
     "cylinders": 8, "displacement": 307.0, "horsepower": 130.0,
     "weight": 3504.0, "acceleration": 12.0, "model_year": 70, "origin": 1},
    {"car_name": "typical 4-cyl compact (1978)",
     "cylinders": 4, "displacement": 120.0, "horsepower": 88.0,
     "weight": 2130.0, "acceleration": 14.5, "model_year": 78, "origin": 2},
    {"car_name": "fuel-sipping 4-cyl (1982)",
     "cylinders": 4, "displacement": 91.0, "horsepower": 67.0,
     "weight": 1995.0, "acceleration": 16.2, "model_year": 82, "origin": 3},
]


def load_model(path=None):
    """Load the pickled best model (joblib)."""
    path = config.BEST_MODEL_PATH if path is None else path
    return joblib.load(path)


def predict_mpg(model, specs_df: pd.DataFrame) -> pd.DataFrame:
    """Return the input frame with an added ``predicted_mpg`` column."""
    missing = [c for c in config.FEATURES if c not in specs_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    preds = model.predict(specs_df[config.FEATURES])
    out = specs_df.copy()
    out["predicted_mpg"] = preds
    return out


def parse_specs(pairs: list) -> pd.DataFrame:
    """Parse ``['cylinders=4', ...]`` into a one-row DataFrame."""
    if not pairs:
        raise ValueError("--specs requires at least one key=value pair")
    row = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected key=value, got: {pair!r}")
        key, value = pair.split("=", 1)
        if key not in config.FEATURES:
            raise ValueError(
                f"Unknown feature {key!r}; choose from {config.FEATURES}"
            )
        row[key] = float(value)
    return pd.DataFrame([row])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict car mileage (mpg) with the trained best model."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--specs", nargs="+", metavar="KEY=VALUE",
                       help="car specs, e.g. cylinders=4 weight=2130 ...")
    group.add_argument("--csv", type=str, help="CSV file with feature columns")
    args = parser.parse_args(argv)

    model = load_model()

    if args.specs:
        frames = [predict_mpg(model, parse_specs(args.specs))]
    elif args.csv:
        cars = pd.read_csv(args.csv)
        frames = [predict_mpg(model, cars)]
    else:
        print("No input given -> running built-in example cars.\n")
        frames = [
            predict_mpg(model, pd.DataFrame([car])) for car in EXAMPLE_CARS
        ]

    print("=" * 70)
    print("MILEAGE PREDICTIONS (best saved model)")
    print("=" * 70)
    for frame in frames:
        cols = ["car_name"] + config.FEATURES + ["predicted_mpg"]
        cols = [c for c in cols if c in frame.columns]
        for _, r in frame[cols].iterrows():
            name = r.get("car_name", "car")
            print(f"  {name}")
            print("    specs:", {c: r[c] for c in config.FEATURES})
            print(f"    --> predicted mpg: {r['predicted_mpg']:.2f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
