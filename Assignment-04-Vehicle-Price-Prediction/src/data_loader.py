"""Load and clean the CarDekho dataset (mirrors Assignment 3 decisions)."""

from __future__ import annotations

import shutil
import urllib.request

import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import PARSED_NUM, parse_features

FALLBACK_URL = ("https://raw.githubusercontent.com/imanishshahu/"
                "Car-Price-Prediction/main/Car%20details%20v3.csv")

MIN_PLAUSIBLE_KM_DRIVEN = 100.0
ZERO_AS_MISSING_COLUMNS = ["mileage_kmpl", "max_power_bhp"]


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV; download the public mirror copy if missing."""
    path = config.RAW_DATA_PATH if path is None else path
    if not path.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[data_loader] {path.name} not found -> downloading mirror")
        with urllib.request.urlopen(FALLBACK_URL, timeout=60) as resp, \
                path.open("wb") as out:
            shutil.copyfileobj(resp, out)
    return pd.read_csv(path)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple:
    """Remove exact duplicate rows; returns (deduped, n_removed)."""
    n_before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, n_before - len(deduped)


def apply_domain_checks(df: pd.DataFrame) -> tuple:
    """Set impossible automotive values to NaN (repaired by imputation later).

    * used car with km_driven < 100 -> data-entry error -> NaN
    * mileage_kmpl == 0, max_power_bhp == 0 -> physically impossible -> NaN

    Returns (repaired_frame, n_values_set_to_nan).
    """
    out = df.copy()
    n_fixed = 0
    mask = out["km_driven"] < MIN_PLAUSIBLE_KM_DRIVEN
    n_fixed += int(mask.sum())
    out.loc[mask, "km_driven"] = np.nan
    for col in ZERO_AS_MISSING_COLUMNS:
        zero_mask = out[col] == 0
        n_fixed += int(zero_mask.sum())
        out.loc[zero_mask, col] = np.nan
    return out, n_fixed


def load_and_prepare(verbose: bool = True) -> pd.DataFrame:
    """Full cleaning path: raw -> parse -> dedup -> domain checks.

    Missing-value imputation is deliberately NOT done here; it lives inside
    the sklearn pipeline where it is fitted on the training split only.
    """
    raw = load_raw()
    if verbose:
        print(f"[data_loader] raw: {raw.shape[0]} rows x {raw.shape[1]} cols")

    parsed = parse_features(raw)
    if verbose:
        miss = parsed[list(PARSED_NUM.values())].isna().sum().sum()
        print(f"[data_loader] parsed string numerics "
              f"({miss} cells missing from raw file)")

    deduped, n_dupes = drop_exact_duplicates(parsed)
    if verbose:
        print(f"[data_loader] removed {n_dupes} exact duplicate rows "
              f"-> {len(deduped)} rows")

    checked, n_fixed = apply_domain_checks(deduped)
    if verbose:
        print(f"[data_loader] domain checks: {n_fixed} impossible values "
              f"-> NaN for imputation")
    return checked
