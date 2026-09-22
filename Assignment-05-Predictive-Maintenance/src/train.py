"""End-to-end training: load -> features -> split -> train -> select -> save.

Run with:  python -m src.train
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_and_prepare
from src.evaluation import (classification_metrics, error_analysis,
                            plot_class_distribution, plot_confusion_matrix,
                            plot_model_comparison, plot_pr_curves,
                            plot_roc_curves, select_best)
from src.feature_engineering import build_feature_frame
from src.models import build_models, build_pipeline


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREDICTIVE MAINTENANCE - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load + validate + dedup ------------------------------------------
    df = load_and_prepare(verbose=True)

    # 2. Feature engineering (row-wise, leak-safe) -------------------------
    X, y = build_feature_frame(df)
    print(f"\nFeatures ({X.shape[1]}): {list(X.columns)}")

    plot_class_distribution(y, config.PLOTS_DIR / "class_distribution.png")

    # 3. Stratified split (BEFORE any learned preprocessing) ---------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y)
    print(f"Train: {X_train.shape[0]} rows ({y_train.sum()} failures) | "
          f"Test: {X_test.shape[0]} rows ({y_test.sum()} failures)")

    # 4-6. Train each model and evaluate with imbalance-aware metrics ------
    results = []
    fitted = {}
    prob_dict = {}
    for name, estimator in build_models().items():
        print(f"\n--- {name} ---")
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        m = classification_metrics(y_test, y_prob)
        m["model"] = name
        results.append(m)
        fitted[name] = pipe
        prob_dict[name] = y_prob
        print(f"  Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['average_precision']:.4f}")

    comparison = (pd.DataFrame(results)
                  .set_index("model")
                  .sort_values(config.SELECTION_METRIC, ascending=False))
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (failure class = positive, threshold=0.5)")
    print("=" * 70)
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

    # 7. Select with the predefined criterion -------------------------------
    best_name = select_best(comparison)
    best_model = fitted[best_name]
    best_prob = prob_dict[best_name]
    print(f"\nSelected model: {best_name} "
          f"(criterion: best {config.SELECTION_METRIC}, tie-breakers "
          f"{config.SELECTION_TIEBREAKERS})")

    # 8. Error analysis for the selected model ------------------------------
    errors = error_analysis(X_test, y_test, best_prob)
    cm_row = comparison.loc[best_name]
    print(f"\nError analysis ({best_name}):")
    print(f"  Confusion: TP={int(cm_row['true_positives'])}  "
          f"FP={int(cm_row['false_positives'])}  "
          f"FN={int(cm_row['false_negatives'])}  "
          f"TN={int(cm_row['true_negatives'])}")
    print(f"  False negatives: {errors['n_false_negatives']} "
          f"(mean predicted failure prob "
          f"{errors['fn_probability_stats']['mean']})")
    if errors["n_false_negatives"]:
        print("  Example false negatives (first 8):")
        print(errors["false_negatives"].to_string(index=False))

    # 9-10. Save model, metadata, metrics, comparison -----------------------
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    metrics_payload = {
        "selected_model": best_name,
        "selection_criterion": (
            f"highest failure-class {config.SELECTION_METRIC} on the test "
            f"set at threshold {config.DECISION_THRESHOLD}; tie-breakers: "
            f"{config.SELECTION_TIEBREAKERS}"),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "target": config.TARGET,
        "excluded_leakage_columns": config.LEAKAGE_COLUMNS,
        "excluded_id_columns": config.ID_COLUMNS,
        "features": config.FEATURES,
        "engineered_features": config.ENGINEERED_FEATURES,
        "class_distribution_full": {
            str(k): int(v)
            for k, v in df[config.TARGET].value_counts().sort_index().items()},
        "models": results,
        "error_analysis": {
            "n_false_positives": errors["n_false_positives"],
            "n_false_negatives": errors["n_false_negatives"],
            "failure_recall": errors["failure_recall"],
            "fn_probability_stats": errors["fn_probability_stats"],
        },
    }
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(metrics_payload, fh, indent=2)
    comparison.to_csv(config.COMPARISON_CSV_PATH)
    with open(config.MODEL_META_PATH, "w") as fh:
        json.dump({
            "selected_model": best_name,
            "features": config.FEATURES,
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "target": config.TARGET,
            "decision_threshold": config.DECISION_THRESHOLD,
            "random_state": config.RANDOM_STATE,
            "pipeline": "preprocessor (impute/scale/onehot) -> classifier",
        }, fh, indent=2)

    # 11. Plots --------------------------------------------------------------
    plot_confusion_matrix(y_test, best_prob,
                          config.PLOTS_DIR / "confusion_matrix.png")
    plot_roc_curves(y_test, prob_dict, config.PLOTS_DIR / "roc_curve.png")
    plot_pr_curves(y_test, prob_dict,
                   config.PLOTS_DIR / "precision_recall_curve.png")
    plot_model_comparison(comparison,
                          config.PLOTS_DIR / "model_comparison.png")

    # 12. Summary ------------------------------------------------------------
    best = comparison.loc[best_name]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best model   : {best_name}")
    print(f"Accuracy     : {best['accuracy']:.4f}")
    print(f"Precision    : {best['precision']:.4f}")
    print(f"Recall       : {best['recall']:.4f}")
    print(f"F1 (failure) : {best['f1']:.4f}")
    print(f"ROC-AUC      : {best['roc_auc']:.4f}")
    print(f"PR-AUC       : {best['average_precision']:.4f}")
    print(f"Artifacts    : {config.ARTIFACTS_DIR}")
    print(f"Saved model  : {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
