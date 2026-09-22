"""Evaluation: metrics in original rupee units, CV, importance, error plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_validate

from src import config
from src.models import get_pipeline

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Metrics (always price space)
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / R^2 in original selling-price units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(1 - np.sum((y_true - y_pred) ** 2)
                    / np.sum((y_true - np.mean(y_true)) ** 2)),
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold shuffled CV on TRAINING data; R^2 in price space."""
    kfold = KFold(n_splits=config.CV_FOLDS, shuffle=True,
                  random_state=config.RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=kfold, scoring=("r2",),
                            n_jobs=-1)
    return {"CV_R2_mean": float(scores["test_r2"].mean()),
            "CV_R2_std": float(scores["test_r2"].std())}


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------
def feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series,
                       max_features: int = 15) -> pd.DataFrame:
    """Permutation importance on the test set (works for every model type).

    Scores the OUTER model (rupee-space predictions via the TTR wrapper)
    and reports importance per ORIGINAL engineered feature column -- no
    need to map one-hot columns back, which keeps the output interpretable.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=config.RANDOM_STATE,
        scoring="neg_mean_absolute_error", n_jobs=-1)
    imp = pd.DataFrame({
        "feature": list(X_test.columns),
        "importance_mae": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return (imp.sort_values("importance_mae", ascending=False)
            .head(max_features).reset_index(drop=True))


def plot_feature_importance(imp_df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    order = imp_df.sort_values("importance_mae")
    ax.barh(order["feature"], order["importance_mae"],
            xerr=order["importance_std"], color="#1f77b4")
    ax.set_xlabel("Permutation importance (increase in MAE when shuffled, INR)")
    ax.set_title("Feature importance - final model (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_target_distribution(y: pd.Series, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(y, bins=60, ax=axes[0], color="#1f77b4")
    axes[0].set_title("selling_price distribution (raw)")
    axes[0].set_xlabel("Price (INR)")
    sns.histplot(np.log1p(y), bins=60, ax=axes[1], color="#d95f02")
    axes[1].set_title("log1p(selling_price) distribution")
    axes[1].set_xlabel("log1p(price)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    lo = 0
    hi = float(max(np.max(y_true), np.max(y_pred))) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, label="Perfect prediction")
    ax.scatter(y_true, y_pred, s=18, alpha=0.45, color="#1f77b4",
               edgecolor="none")
    ax.set_xlabel("Actual price (INR)")
    ax.set_ylabel("Predicted price (INR)")
    ax.set_title("Actual vs Predicted selling price (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, y_pred, out_path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, s=18, alpha=0.45, color="#d95f02",
               edgecolor="none")
    ax.axhline(0, color="black", ls="--", lw=1.4)
    ax.set_xlabel("Predicted price (INR)")
    ax.set_ylabel("Residual (actual - predicted, INR)")
    ax.set_title("Residual plot (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_df: pd.DataFrame, out_path) -> None:
    metrics = [("Test_MAE", "Test MAE (lower better, INR)"),
               ("Test_RMSE", "Test RMSE (lower better, INR)"),
               ("Test_R2", "Test R^2 (higher better)"),
               ("CV_R2_mean", "CV R^2 (higher better)")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(data=comparison_df.reset_index(), x="model", y=metric,
                    ax=ax, color="#4c9be8")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Model comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_price_range(y_true, y_pred, out_path,
                              bins: int = 6) -> None:
    """Mean absolute error per actual-price decile-ish bucket."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_true - y_pred)
    buckets = pd.qcut(pd.Series(y_true), q=bins, duplicates="drop")
    frame = pd.DataFrame({"range": buckets, "abs_err": abs_err})
    agg = frame.groupby("range", observed=True)["abs_err"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(agg)), agg.values, color="#2ca02c")
    labels = [f"{int(iv.left/1e5)}L-{int(np.ceil(iv.right/1e5))}L"
              for iv in agg.index]
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel("Actual price range (L = lakh INR)")
    ax.set_ylabel("Mean absolute error (INR)")
    ax.set_title("Prediction error by actual price range (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return agg


def largest_errors(X_test: pd.DataFrame, y_true, y_pred,
                   n: int = 8) -> pd.DataFrame:
    """Return the n worst test predictions with car context."""
    frame = X_test.copy()
    frame["actual_price"] = np.asarray(y_true)
    frame["predicted_price"] = np.asarray(y_pred)
    frame["abs_error"] = np.abs(frame["actual_price"]
                                - frame["predicted_price"])
    cols = ["brand", "vehicle_age", "km_driven", "fuel", "transmission",
            "actual_price", "predicted_price", "abs_error"]
    cols = [c for c in cols if c in frame.columns]
    return frame.nlargest(n, "abs_error")[cols].reset_index(drop=True)
