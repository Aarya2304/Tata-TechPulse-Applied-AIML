"""Exploratory Data Analysis (EDA): summary statistics and diagnostic plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: save files, no display needed

import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data_pipeline import missing_value_report

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------
def print_overview(df) -> None:
    print("=" * 70)
    print("EDA - OVERVIEW")
    print("=" * 70)
    print(f"Shape (rows, cols): {df.shape}")
    print(f"Duplicates removed during cleaning: {df.attrs.get('duplicates_removed', 0)}")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column (non-zero only):")
    report = missing_value_report(df)
    print(report.to_string() if len(report) else "  none")
    print("\nTarget (mpg) summary:")
    print(df[config.TARGET].describe().to_string())


def print_correlations(df) -> None:
    corr = df[config.FEATURES + [config.TARGET]].corr(numeric_only=True)
    print("\nCorrelation of each feature with mpg:")
    print(corr[config.TARGET].drop(config.TARGET).sort_values().to_string())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_distributions(df, out_path) -> None:
    """Histograms + KDE for the target and every model feature."""
    cols = [config.TARGET] + config.FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#3776ab")
        ax.set_title(col, fontsize=14)
        ax.set_xlabel("")
    fig.suptitle("Auto MPG - Feature & Target Distributions", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df, out_path) -> None:
    corr = df[config.FEATURES + [config.TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (Pearson)", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_vs_target(df, out_path) -> None:
    """Scatter plots of mpg against the most informative features."""
    cols = ["weight", "horsepower", "displacement", "cylinders",
            "acceleration", "model_year"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, col in zip(axes.ravel(), cols):
        sns.scatterplot(data=df, x=col, y=config.TARGET, ax=ax,
                        hue="origin", palette="deep", alpha=0.75, legend=False)
        ax.set_title(f"mpg vs {col}", fontsize=14)
    fig.suptitle("mpg vs Key Features (hue = origin: 1=USA, 2=Europe, 3=Japan)",
                 fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mpg_by_category(df, out_path) -> None:
    """Box plots: mpg by cylinder count and by origin."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df, x="cylinders", y=config.TARGET, ax=axes[0],
                color="#6baed6")
    axes[0].set_title("mpg by cylinder count")
    sns.boxplot(data=df, x="origin", y=config.TARGET, ax=axes[1],
                color="#fdae6b")
    axes[1].set_title("mpg by origin (1=USA, 2=Europe, 3=Japan)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(df, plots_dir=None, verbose: bool = True) -> None:
    """Run the full EDA: console summaries + all plots saved to disk."""
    plots_dir = config.PLOTS_DIR if plots_dir is None else plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print_overview(df)
        print_correlations(df)
        print("\nEDA plots ->", plots_dir)

    plot_distributions(df, plots_dir / "eda_distributions.png")
    plot_correlation_heatmap(df, plots_dir / "eda_correlation_heatmap.png")
    plot_scatter_vs_target(df, plots_dir / "eda_scatter_vs_mpg.png")
    plot_mpg_by_category(df, plots_dir / "eda_mpg_by_category.png")

    if verbose:
        for name in [
            "eda_distributions.png",
            "eda_correlation_heatmap.png",
            "eda_scatter_vs_mpg.png",
            "eda_mpg_by_category.png",
        ]:
            print(f"  saved {plots_dir / name}")
