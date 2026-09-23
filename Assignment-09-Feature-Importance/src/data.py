"""Data acquisition, unit parsing, feature engineering and splitting.

Dataset provenance
------------------
CarDekho "Car details v3" (8,128 rows x 13 columns) — the same raw CSV used
by Assignment 3 (cleaning) and Assignment 4 (price prediction) in this
repository. It is located on disk and opened **read-only**; nothing is copied
into or written back to those assignments. If neither local copy exists (e.g.
a fresh clone), the file is downloaded from the public GitHub mirror and
stored inside this assignment's own data/ directory instead.

Kaggle origin: nehalbirla/vehicle-dataset-from-cardekho ("Car details v3.csv").
"""

from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def locate_raw_csv() -> Path:
    """Return a path to the raw CarDekho CSV, downloading if necessary.

    Order: local sibling copies (opened read-only) → this assignment's data/
    cache (downloaded). The returned path is never written to by this project.
    """
    for candidate in config.SIBLING_DATASET_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > 100_000:
            return candidate
    local = config.DATA_DIR / "car_data.csv"
    if local.is_file() and local.stat().st_size > 100_000:
        return local
    print(f"Local dataset copies not found; downloading from {config.DATASET_URL} ...")
    local.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        import subprocess

        subprocess.run([curl, "-sS", "-L", "-o", str(local), config.DATASET_URL], check=True)
    else:  # pragma: no cover - fallback path
        urllib.request.urlretrieve(config.DATASET_URL, local)
    return local


def dataset_provenance(path: Path) -> dict:
    """Describe where the data came from (for metrics.json and the README)."""
    return {
        "dataset": config.DATASET_NAME,
        "path_used": str(Path(path).name),
        "source_type": "reused read-only from Assignment 3/4 local copy"
        if any(Path(path).resolve() == c.resolve() for c in config.SIBLING_DATASET_CANDIDATES)
        else "downloaded to this assignment's data/ directory",
        "download_fallback_url": config.DATASET_URL,
        "origin": "Kaggle: nehalbirla/vehicle-dataset-from-cardekho (Car details v3)",
    }


# ---------------------------------------------------------------------------
# Unit parsing (safe against unexpected formats)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _extract_number(value, default=np.nan) -> float:
    """First numeric token of a string like '23.4 kmpl' → 23.4."""
    if pd.isna(value):
        return default
    m = _NUM_RE.search(str(value))
    return float(m.group()) if m else default


def parse_mileage(series: pd.Series) -> pd.Series:
    """'23.4 kmpl' / '23.4 km/kg' → 23.4 (both are km-per-unit fuel)."""
    return series.map(lambda v: _extract_number(v))


def parse_engine(series: pd.Series) -> pd.Series:
    """'1248 CC' → 1248.0."""
    return series.map(lambda v: _extract_number(v))


def parse_max_power(series: pd.Series) -> pd.Series:
    """'74 bhp' → 74.0."""
    return series.map(lambda v: _extract_number(v))


def parse_torque(series: pd.Series) -> pd.Series:
    """Parse torque strings and convert kgm → Nm (1 kgm = 9.80665 Nm).

    Formats observed in the dataset:
      '190Nm@ 2000rpm', '250Nm@ 1500-2500rpm', '12.7 kgm@ 2500 rpm'
    The first number is the torque magnitude; the unit determines conversion.
    """
    def parse(value) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).lower()
        m = _NUM_RE.search(text)
        if not m:
            return np.nan
        torque = float(m.group())
        if "kgm" in text:
            torque *= 9.80665
        return torque

    return series.map(parse)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
REFERENCE_YEAR = 2020  # documented fixed reference (dataset's latest model year)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse string-typed numerics and add engineered features.

    Returns a new frame with the original columns preserved plus:
    mileage_kmpl, engine_cc, max_power_bhp, torque_nm, vehicle_age, km_per_year, brand.
    """
    out = df.copy()
    out["mileage_kmpl"] = parse_mileage(df["mileage"])
    out["engine_cc"] = parse_engine(df["engine"])
    out["max_power_bhp"] = parse_max_power(df["max_power"])
    out["torque_nm"] = parse_torque(df["torque"])
    out["vehicle_age"] = REFERENCE_YEAR - df["year"]
    out["km_per_year"] = df["km_driven"] / out["vehicle_age"].replace(0, np.nan)
    out["brand"] = df["name"].astype("string").str.strip().str.split().str[0]
    return out


NUMERIC_FEATURES = (
    config.RAW_NUMERIC_FEATURES
    + config.PARSED_NUMERIC_FEATURES
    + config.ENGINEERED_NUMERIC_FEATURES
)
CATEGORICAL_FEATURES = config.RAW_CATEGORICAL_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Load + prepare
# ---------------------------------------------------------------------------


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load the raw CSV, engineer features, return (X, y, provenance).

    - drops rows with a missing target (3 in the raw file);
    - never includes the target in the feature matrix;
    - duplicates are kept (they are genuine repeated listings, and dropping
      them is a preprocessing choice made in Assignment 3/4 — not needed to
      study feature importance).
    """
    path = locate_raw_csv()
    provenance = dataset_provenance(path)
    raw = pd.read_csv(path)
    provenance["rows_raw"] = int(len(raw))

    df = raw.dropna(subset=[config.TARGET]).copy()
    provenance["rows_dropped_missing_target"] = int(len(raw) - len(df))
    df = engineer_features(df)

    y = df[config.TARGET].astype(float)
    X = df[FEATURES].copy()
    provenance["rows_final"] = int(len(X))
    provenance["n_features"] = len(FEATURES)
    provenance["numeric_features"] = NUMERIC_FEATURES
    provenance["categorical_features"] = CATEGORICAL_FEATURES
    return X, y, provenance


def make_splits(
    X: pd.DataFrame, y: pd.Series, random_state: int = config.RANDOM_STATE
):
    """Reproducible train/test split (80/20, seed 42)."""
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=random_state)
