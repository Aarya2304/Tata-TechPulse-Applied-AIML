"""Prediction CLI: load the saved pipeline and price one vehicle or a CSV.

Usage:
    python -m src.predict                 # built-in demo vehicles
    python -m src.predict --csv file.csv  # batch predictions from a CSV

The CSV must contain the engineered feature columns (name, year, km_driven,
fuel, seller_type, transmission, owner, mileage, engine, max_power, torque,
seats - raw CarDekho-style columns are fine); they are re-engineered here
using the same functions as training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.feature_engineering import build_features

# Raw CarDekho-style demo vehicles (same schema as the Kaggle dataset).
DEMO_VEHICLES = [
    {"name": "Maruti Swift VDI", "year": 2019, "km_driven": 32000,
     "fuel": "Diesel", "seller_type": "Individual",
     "transmission": "Manual", "owner": "First Owner",
     "mileage": "25.8 kmpl", "engine": "1248 CC", "max_power": "74 bhp",
     "torque": "190Nm@ 2000rpm", "seats": 5.0},
    {"name": "Hyundai i10 Magna", "year": 2015, "km_driven": 58000,
     "fuel": "Petrol", "seller_type": "Individual",
     "transmission": "Manual", "owner": "Second Owner",
     "mileage": "19.77 kmpl", "engine": "998 CC", "max_power": "68.05 bhp",
     "torque": "90Nm@ 3500rpm", "seats": 5.0},
    {"name": "Toyota Innova 2.5 GX", "year": 2012, "km_driven": 120000,
     "fuel": "Diesel", "seller_type": "Dealer", "transmission": "Manual",
     "owner": "Third Owner", "mileage": "13.5 kmpl", "engine": "2494 CC",
     "max_power": "100 bhp", "torque": "200Nm@ 1400rpm", "seats": 8.0},
]


def predict_frame(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for raw rows and predict prices in rupees."""
    feats = build_features(raw_df)
    feats = feats.drop(columns=[config.TARGET],
                       errors="ignore")  # never accept a provided target
    feats = feats[config.FEATURES]
    prices = model.predict(feats)
    return pd.DataFrame({"predicted_price_inr": prices})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict used-vehicle selling price with the saved model.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV with raw vehicle rows to score.")
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
        preds.insert(0, "name", raw.get("name", ""))
        out = path.with_name(path.stem + "_predictions.csv")
        preds.to_csv(out, index=False)
        print(f"\nWrote {len(preds)} predictions to {out}")
        print(preds.to_string(index=False))
        return

    # Demo mode ---------------------------------------------------------
    demo = pd.DataFrame(DEMO_VEHICLES)
    preds = predict_frame(model, demo)
    print("\nDemo vehicles (actual saved model, no hardcoded values):\n")
    for row, price in zip(DEMO_VEHICLES, preds["predicted_price_inr"]):
        # 'Rs.' instead of the rupee glyph: Windows cp1252 consoles
        # cannot encode U+20B9.
        print(f"  {row['year']} {row['name']:<28}"
              f" -> Predicted vehicle price: Rs.{price:,.0f}")


if __name__ == "__main__":
    main()
