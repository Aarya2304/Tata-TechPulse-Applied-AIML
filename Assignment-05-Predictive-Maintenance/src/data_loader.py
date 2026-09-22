"""Load, validate and inspect the AI4I 2020 predictive maintenance data."""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile

import pandas as pd

from src import config

UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/601/"
               "ai4i+2020+predictive+maintenance+dataset.zip")


def _download_official_zip(dest_csv) -> None:
    """Download the official UCI zip and extract the CSV into data/."""
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data_loader] {dest_csv.name} not found -> downloading "
          f"official UCI copy")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(member) as src, dest_csv.open("wb") as out:
            shutil.copyfileobj(src, out)


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV, auto-downloading the official dataset if absent."""
    path = config.RAW_DATA_PATH if path is None else path
    if not path.exists():
        _download_official_zip(path)
    df = pd.read_csv(path)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise a clear error if required columns/values are missing."""
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if not df[config.TARGET].isin([0, 1]).all():
        raise ValueError(f"Target {config.TARGET!r} must be binary 0/1")


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows; returns (deduped, n_removed)."""
    n_before = len(df)
    deduped = df.drop_duplicates().reset_index(drop=True)
    return deduped, n_before - len(deduped)


def target_distribution(df: pd.DataFrame) -> pd.Series:
    """Counts of each target class (documented in README/metrics)."""
    return df[config.TARGET].value_counts().sort_index()


def load_and_prepare(verbose: bool = True) -> pd.DataFrame:
    """load -> validate -> dedup. Returns the cleaned raw frame.

    No imputation or scaling happens here: the dataset has no missing
    values, and any learned preprocessing lives inside the sklearn
    pipeline so it is fitted on the training split only.
    """
    df = load_raw()
    validate(df)
    if verbose:
        print(f"[data_loader] raw: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"[data_loader] missing cells: {int(df.isna().sum().sum())}")

    df, n_dupes = drop_duplicates(df)
    if verbose:
        print(f"[data_loader] removed {n_dupes} exact duplicate rows "
              f"-> {len(df)} rows")
        dist = target_distribution(df)
        share = dist.get(1, 0) / len(df) * 100
        print(f"[data_loader] target distribution: "
              f"0={dist.get(0, 0)}  1={dist.get(1, 0)} "
              f"({share:.2f}% failures)")
    return df
