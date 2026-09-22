"""Data profiling: describe the raw dataset BEFORE any modification.

Produces:
* ``artifacts/reports/data_profile.json``   (machine-readable profile)
* ``artifacts/reports/profile_numeric_summary.csv``
* ``artifacts/reports/profile_categorical_summary.csv``
* ``artifacts/plots/missing_values_before.png``
"""

from __future__ import annotations

import json

import pandas as pd

from src import config


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Build the full per-column profile of a dataframe."""
    profile = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "per_column": {},
    }
    for col in df.columns:
        s = df[col]
        entry = {
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 3),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            entry["numeric_summary"] = {k: round(float(v), 4)
                                        for k, v in desc.items()}
            entry["zeros_count"] = int((s == 0).sum())
            entry["negative_count"] = int((s < 0).sum())
        else:
            vc = s.value_counts(dropna=True)
            entry["top_values"] = {str(k): int(v)
                                   for k, v in vc.head(8).items()}
        profile["per_column"][col] = entry
    return profile


def numeric_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    return num.describe().T.round(2)


def categorical_summary_frame(df: pd.DataFrame, max_values: int = 6) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(exclude="number").columns:
        vc = df[col].value_counts(dropna=True)
        top = ", ".join(f"{k} ({v})" for k, v in vc.head(max_values).items())
        rows.append({
            "column": col,
            "unique": df[col].nunique(dropna=True),
            "missing": int(df[col].isna().sum()),
            f"top_{max_values}_values": top,
        })
    return pd.DataFrame(rows)


def save_profile(df: pd.DataFrame, out_path=None) -> dict:
    out_path = config.DATA_PROFILE_PATH if out_path is None else out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile_dataframe(df)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def save_missing_plot(df: pd.DataFrame, out_path) -> None:
    """Bar chart of missing values per column (columns with any missing)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                fontsize=14)
        ax.set_axis_off()
    else:
        ax.bar(miss.index, miss.values, color="#d95f02")
        for i, v in enumerate(miss.values):
            ax.text(i, v + miss.max() * 0.01, str(int(v)), ha="center",
                    fontsize=10)
        ax.set_ylabel("Missing values (count)")
        ax.set_title("Missing values BEFORE cleaning (raw data)")
        ax.set_ylim(0, miss.max() * 1.15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_profiling(df: pd.DataFrame) -> dict:
    """Save profile JSON, CSV summaries and the before-plot. Returns profile."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    profile = save_profile(df)
    numeric_summary_frame(df).to_csv(
        config.REPORTS_DIR / "profile_numeric_summary.csv")
    categorical_summary_frame(df).to_csv(
        config.REPORTS_DIR / "profile_categorical_summary.csv", index=False)
    save_missing_plot(df, config.PLOTS_DIR / "missing_values_before.png")
    return profile
