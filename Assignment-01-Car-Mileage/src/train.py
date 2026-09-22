"""Training pipeline for the Auto MPG regression assignment.

Steps
-----
1. Load & clean the UCI Auto MPG data.
2. Stratified train/test split (80/20, fixed seed) on binned MPG so both
   splits see the same fuel-economy range -- leakage-free by construction.
3. Three candidate models, each a full scikit-learn Pipeline:
   Linear Regression, Ridge Regression, Random Forest Regressor.
   The leakage-free ``HorsepowerImputer`` lives inside every pipeline, so it
   is re-fit on the training folds of each CV split automatically.
4. 5-fold cross-validation on the training set -> model comparison table.
5. Refit each candidate on the full training set, evaluate once on the
   untouched test set (MAE / RMSE / R^2).
6. Save all plots, the comparison table, and the best model via joblib.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_pipeline import HorsepowerImputer, load_and_clean
from src.eda import run_eda
from src.metrics_utils import cross_validate_model, metrics_to_frame, regression_metrics

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Data split (leakage-free)
# ---------------------------------------------------------------------------
def make_split(df: pd.DataFrame):
    """Stratified 80/20 split on binned mpg; returns (train_df, test_df)."""
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    # Bin mpg into 5 quantile buckets -> stratification labels.
    strata = pd.qcut(df[config.TARGET], q=5, labels=False, duplicates="drop")
    train_idx, test_idx = next(splitter.split(df, strata))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def build_models() -> dict:
    """Return name -> sklearn Pipeline for the three candidate models."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("imputer", HorsepowerImputer()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            steps=[
                ("imputer", HorsepowerImputer()),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=config.RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", HorsepowerImputer()),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_on_test(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    m = regression_metrics(y_test, preds)
    return {"model": name, **m, "predictions": preds}


def cross_validate_all(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> list:
    rows = []
    for name, model in models.items():
        cv = cross_validate_model(model, X_train, y_train)
        rows.append({
            "model": name,
            "CV_MAE_mean": cv["MAE"]["mean"], "CV_MAE_std": cv["MAE"]["std"],
            "CV_RMSE_mean": cv["RMSE"]["mean"], "CV_RMSE_std": cv["RMSE"]["std"],
            "CV_R2_mean": cv["R2"]["mean"], "CV_R2_std": cv["R2"]["std"],
        })
        print(f"  {name}: CV R2 = {cv['R2']['mean']:.3f} "
              f"(+/- {cv['R2']['std']:.3f})")
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(y_test, y_pred, out_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    lo = float(min(y_test.min(), y_pred.min())) - 1
    hi = float(max(y_test.max(), y_pred.max())) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
    ax.scatter(y_test, y_pred, alpha=0.65, edgecolor="white", s=60,
               color="#3776ab", label="Test cars")
    ax.set_xlabel("Actual mpg")
    ax.set_ylabel("Predicted mpg")
    ax.set_title("Actual vs Predicted mpg (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_test, y_pred, out_path) -> None:
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.65, edgecolor="white", s=60,
               color="#d95f02")
    ax.axhline(0, color="black", ls="--", lw=1.5)
    ax.set_xlabel("Predicted mpg")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual Plot (Test Set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_df: pd.DataFrame, out_path) -> None:
    metrics = ["CV_MAE_mean", "CV_RMSE_mean", "CV_R2_mean"]
    pretty = {"CV_MAE_mean": "CV MAE (lower better)",
              "CV_RMSE_mean": "CV RMSE (lower better)",
              "CV_R2_mean": "CV R^2 (higher better)"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=comparison_df.reset_index(), x="model",
                    y=metric, ax=ax, color="#4c9be8")
        ax.set_title(pretty[metric], fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Model Comparison - 5-fold Cross-Validation", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> dict:
    print("=" * 70)
    print("STEP 1/5  Load & clean data")
    df = load_and_clean()
    print(f"  rows after cleaning: {len(df)}")

    print("STEP 2/5  EDA")
    run_eda(df, verbose=False)
    print("  EDA plots saved")

    print("STEP 3/5  Train/test split (stratified 80/20)")
    train_df, test_df = make_split(df)
    X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]
    X_test, y_test = test_df[config.FEATURES], test_df[config.TARGET]
    print(f"  train={len(train_df)} rows, test={len(test_df)} rows")

    print("STEP 4/5  Cross-validate 3 candidate models (5-fold, training set)")
    models = build_models()
    cv_rows = cross_validate_all(models, X_train, y_train)

    print("STEP 5/5  Refit on full training set & evaluate on test set")
    test_rows, preds_by_model = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        row = evaluate_on_test(name, model, X_test, y_test)
        test_rows.append({k: v for k, v in row.items() if k != "predictions"})
        preds_by_model[name] = row["predictions"]
        print(f"  {name}: test R2 = {row['R2']:.3f}")

    comparison_df = metrics_to_frame(cv_rows + test_rows)
    comparison_df.to_csv(config.COMPARISON_CSV_PATH)

    best_name = max(test_rows, key=lambda r: r["R2"])["model"]
    print(f"\nBest model (highest test R^2): {best_name}")

    # Refit best model on the FULL dataset for deployment.
    best_model = models[best_name]
    best_model.fit(df[config.FEATURES], df[config.TARGET])

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_model, config.BEST_MODEL_PATH)

    # Plots for the best model's test predictions.
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred_best = preds_by_model[best_name]
    plot_actual_vs_predicted(y_test, y_pred_best,
                             config.PLOTS_DIR / "actual_vs_predicted.png")
    plot_residuals(y_test, y_pred_best, config.PLOTS_DIR / "residuals.png")
    plot_model_comparison(comparison_df, config.PLOTS_DIR / "model_comparison.png")

    # Persist all metrics for the README.
    metrics_report = {
        "best_model": best_name,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "models": {
            name: {
                "cross_validation": next(
                    {k: v for k, v in r.items() if k != "model"}
                    for r in cv_rows if r["model"] == name),
                "test_set": next(
                    {k: v for k, v in r.items() if k != "model"}
                    for r in test_rows if r["model"] == name),
            }
            for name in models
        },
    }
    config.METRICS_PATH.write_text(json.dumps(metrics_report, indent=2))

    print(f"\nArtifacts written to {config.ARTIFACTS_DIR}")
    print(f"  best model -> {config.BEST_MODEL_PATH.name}")
    print(f"  metrics    -> {config.METRICS_PATH.name}")
    print(f"  comparison -> {config.COMPARISON_CSV_PATH.name}")
    return metrics_report


if __name__ == "__main__":
    main()
