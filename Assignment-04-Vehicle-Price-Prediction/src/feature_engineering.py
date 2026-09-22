"""Feature engineering: unit parsing + structured automotive features.

All parsing/derivation here is row-wise and target-independent: no feature
uses ``selling_price`` or any statistic learned from the data, so nothing in
this module can leak the target.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config

_NM_PER_KGM = 9.80665

# raw column -> engineered numeric column
PARSED_NUM = {
    "mileage": "mileage_kmpl",
    "engine": "engine_cc",
    "max_power": "max_power_bhp",
    "torque": "torque_nm",
}


def parse_numeric_value(value) -> float:
    """Extract the leading float: '23.4 kmpl' -> 23.4, '1248 CC' -> 1248."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else np.nan


def parse_torque(value) -> float:
    """Parse torque strings to Newton-metres.

    Handles the three formats present in the dataset:
      '190Nm@ 2000rpm'            -> 190.0 Nm
      '22.4 kgm at 1750-2750rpm'  -> 22.4 * 9.80665 = 219.7 Nm
      '12.7@ 2,700(kgm@ rpm)'     -> 12.7 * 9.80665 = 124.5 Nm
    """
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    value_num = float(match.group())
    if "kgm" in text:
        value_num *= _NM_PER_KGM
    return value_num


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse unit-suffixed strings into numeric columns + derive brand.

    Raw string columns (name/mileage/engine/max_power/torque) are consumed
    and dropped; ``selling_price`` passes through untouched as the target.
    """
    out = df.copy()
    for src, dst in PARSED_NUM.items():
        if src == "torque":
            out[dst] = out[src].map(parse_torque)
        else:
            out[dst] = out[src].map(parse_numeric_value)

    out["brand"] = out["name"].astype(str).str.strip().str.split().str[0]
    out = out.drop(columns=list(PARSED_NUM) + ["name"])
    return out


def add_vehicle_age(df: pd.DataFrame,
                    reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    """vehicle_age = reference_year - year (reference year is fixed in
    config: 2020, the newest model year in the dataset, so ages are >= 0)."""
    out = df.copy()
    out["vehicle_age"] = reference_year - out["year"].astype(int)
    return out


def add_km_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Usage intensity: km_driven divided by max(age, 1) year."""
    out = df.copy()
    age = out["vehicle_age"].clip(lower=1)
    out["km_per_year"] = out["km_driven"] / age
    return out


def build_features(df: pd.DataFrame,
                   reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    """Full feature engineering: parse -> age -> usage intensity."""
    out = parse_features(df)
    out = add_vehicle_age(out, reference_year)
    out = add_km_per_year(out)
    return out


def prepare_prediction_frame(record: dict) -> pd.DataFrame:
    """Turn a single car spec dict into a one-row feature frame.

    Accepts either raw strings (\"23.4 kmpl\") or plain numbers; ``year`` is
    converted to ``vehicle_age`` with the configured reference year.
    """
    row = dict(record)
    year = row.pop("year", None)
    for src, dst in PARSED_NUM.items():
        if src in row:
            parser = parse_torque if src == "torque" else parse_numeric_value
            row[dst] = parser(row.pop(src))
    if "name" in row:
        row["brand"] = str(row.pop("name")).strip().split()[0]
    if year is not None:
        row["vehicle_age"] = config.REFERENCE_YEAR - int(year)
    if "km_per_year" not in row and row.get("vehicle_age") is not None:
        row["km_per_year"] = (row.get("km_driven", 0.0)
                              / max(row["vehicle_age"], 1))
    frame = pd.DataFrame([row])
    missing = [c for c in config.FEATURES if c not in frame.columns]
    for col in missing:
        frame[col] = np.nan
    return frame[config.FEATURES]
