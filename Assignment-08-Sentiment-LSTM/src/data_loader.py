"""Dataset loading, validation, cleaning and inspection.

Dataset: Amazon Automotive product reviews ("reviews_Automotive_5.json.gz",
20,473 reviews of vehicle parts/accessories), from the SNAP / UCSD McAuley
Amazon product-data release (Julian McAuley, 2016):
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz

The file is downloaded automatically when missing. Real measurements (from
inspecting this exact file):

- 20,473 rows, 9 JSON fields; text field ``reviewText``, rating ``overall``.
- ratings: 1★=542, 2★=606, 3★=1,430, 4★=3,967, 5★=13,928.
- 0 missing reviewText/overall values in the raw JSON (empty strings are
  handled as missing by :func:`clean_dataframe`), 5 duplicate review texts.

Binary sentiment labels are derived as: 1-2★ → negative (0), 3★ → neutral
(dropped, documented), 4-5★ → positive (1). No labels are invented.
"""

from __future__ import annotations

import gzip
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from src import config


def download_dataset(dest: Path | None = None, url: str | None = None) -> Path:
    """Download the reviews archive unless a valid local copy already exists."""
    dest = Path(dest) if dest is not None else config.RAW_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url or config.DATASET_URL
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    print(f"Downloading {config.DATASET_NAME} ...")
    curl = shutil.which("curl")
    if curl:
        import subprocess

        subprocess.run([curl, "-S", "-L", "-o", str(dest), "--retry", "3", url], check=True)
    else:  # pragma: no cover - fallback path
        urllib.request.urlretrieve(url, dest)
    return dest


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Read the gzipped JSON-lines reviews into a DataFrame.

    Keeps the three fields used by the project: review text, review title and
    the 1-5 star rating.
    """
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not path.exists():
        download_dataset(path)
    records = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    config.TEXT_COLUMN: row.get(config.TEXT_COLUMN),
                    config.TITLE_COLUMN: row.get(config.TITLE_COLUMN),
                    config.RATING_COLUMN: row.get(config.RATING_COLUMN),
                }
            )
    return pd.DataFrame(records)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate columns, derive labels, drop missing/duplicate rows.

    Returns the cleaned DataFrame and a dict of measured cleaning statistics
    (nothing is hardcoded; every number is computed from the input).
    """
    required = [config.TEXT_COLUMN, config.RATING_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    stats: dict[str, object] = {"rows_loaded": int(len(df))}

    # --- rating -> binary sentiment label ---------------------------------
    ratings = pd.to_numeric(df[config.RATING_COLUMN], errors="coerce")
    stats["missing_ratings"] = int(ratings.isna().sum())

    text = df[config.TEXT_COLUMN].astype("string")
    stats["missing_text_raw"] = int((text.isna() | (text.fillna("").str.strip() == "")).sum())

    labels = ratings.map(
        lambda s: None if (pd.isna(s) or s == config.NEUTRAL_RATING)
        else (1 if s >= 4 else 0)
    )
    stats["neutral_reviews_dropped"] = int(
        (ratings == config.NEUTRAL_RATING).sum()
    )

    work = pd.DataFrame(
        {
            "text": text,
            "rating": ratings,
            "label": labels,
        }
    )

    # --- drop rows without text or without a usable label -----------------
    valid_text = work["text"].fillna("").str.strip() != ""
    before = len(work)
    work = work[valid_text & work["label"].notna()].copy()
    work["label"] = work["label"].astype(int)  # labels are 0/1 after the drop
    stats["rows_dropped_missing_text_or_label"] = int(before - len(work))

    # --- combine title + text (title carries sentiment signal) ------------
    titles = df.loc[work.index, config.TITLE_COLUMN].astype("string").fillna("")
    work["text"] = (
        titles.fillna("").str.strip() + " . " + work["text"].fillna("").str.strip()
    ).str.strip(" .")

    # --- drop exact duplicate review texts (keep first occurrence) --------
    before = len(work)
    work = work.drop_duplicates(subset=["text"], keep="first")
    stats["duplicates_removed"] = int(before - len(work))

    # --- final bookkeeping -------------------------------------------------
    work = work.reset_index(drop=True)
    stats["rows_final"] = int(len(work))
    stats["class_counts"] = {
        str(k): int(v) for k, v in work["label"].value_counts().sort_index().items()
    }
    stats["class_distribution"] = {
        str(k): round(v / len(work), 4) for k, v in work["label"].value_counts(normalize=True).sort_index().items()
    }
    stats["word_length_stats"] = {
        "min": int(work["text"].str.split().str.len().min()),
        "median": float(work["text"].str.split().str.len().median()),
        "mean": round(float(work["text"].str.split().str.len().mean()), 1),
        "p95": int(work["text"].str.split().str.len().quantile(0.95)),
        "max": int(work["text"].str.split().str.len().max()),
    }
    out = work.rename(columns={"text": config.TEXT_COLUMN, "rating": config.RATING_COLUMN,
                               "label": config.LABEL_COLUMN})
    keep_cols = [config.TEXT_COLUMN, config.RATING_COLUMN, config.LABEL_COLUMN]
    return out[keep_cols], stats


def dataset_report(df: pd.DataFrame, stats: dict) -> str:
    """Human-readable inspection summary used by train.py and the README."""
    lines = [
        f"rows loaded:            {stats['rows_loaded']}",
        f"missing text cells:     {stats['missing_text_raw']}",
        f"missing ratings:        {stats['missing_ratings']}",
        f"neutral (3-star) rows:  {stats['neutral_reviews_dropped']} (dropped)",
        f"rows dropped (no text/label): {stats['rows_dropped_missing_text_or_label']}",
        f"duplicate texts removed: {stats['duplicates_removed']}",
        f"final usable rows:      {stats['rows_final']}",
        f"class counts:           {stats['class_counts']}",
        f"class distribution:     {stats['class_distribution']}",
        f"text length (words):    {stats['word_length_stats']}",
    ]
    return "\n".join(lines)


def load_and_clean(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: download if needed, load, clean, report."""
    df = load_raw_dataframe(path)
    return clean_dataframe(df)
