"""Load the CarDekho used-car dataset (local copy with URL fallback)."""

from __future__ import annotations

import shutil
import urllib.request

import pandas as pd

from src import config

FALLBACK_URL = ("https://raw.githubusercontent.com/imanishshahu/"
                "Car-Price-Prediction/main/Car%20details%20v3.csv")


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV, downloading the public copy if it is missing."""
    path = config.RAW_DATA_PATH if path is None else path
    if not path.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[load_data] {path.name} not found -> downloading from mirror")
        with urllib.request.urlopen(FALLBACK_URL, timeout=60) as resp, \
                path.open("wb") as out:
            shutil.copyfileobj(resp, out)
    df = pd.read_csv(path)
    return df
