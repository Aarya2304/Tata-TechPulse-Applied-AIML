"""End-to-end training: load -> engineer -> split -> train -> compare -> save.

Run with:  python -m src.train
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_and_prepare
from src.evaluation import (
    cross_validate_model,
    feature_importance,
    largest_errors,
    plot_actual_vs_predicted,
    plot_error_by_price_range,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_target_distribution,
    regression_metrics,
)
from src.feature_engineering import add_km_per_year, add_vehicle_age
from src.models import build_models


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load + clean (parse unit strings, dedup, domain checks)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("VEHICLE PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 70)
    df = load_and_prepare(verbose=True)

    # ------------------------------------------------------------------
    # 2. Feature engineering (row-wise, target-independent -> pre-split safe)
    # ------------------------------------------------------------------
    df = add_vehicle_age(df)
    df = add_km_per_year(df)

    target = config.TARGET
    y = df[target]
    X = df[config.FEATURES]          # explicit selection: target excluded

    print(f"\nAfter feature engineering: {X.shape[1]} features, "
          f"{len(df)} rows")
    print(f"Target stats (INR): min={y.min():,.0f}  median={y.median():,.0f}"
          f"  max={y.max():,.0f}  skew={y.skew():.2f}")

    plot_target_distribution(y, config.PLOTS_DIR / "price_distribution.png")

    # ------------------------------------------------------------------
    # 3. Split BEFORE any statistics-learned preprocessing
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ------------------------------------------------------------------
    # 4-7. Train + evaluate every model (log1p target inside TTR wrapper;
    #      predictions come back in rupees via expm1)
    # ------------------------------------------------------------------
    results = []
    fitted = {}
    models = build_models()
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred_price = model.predict(X_test)

        m = regression_metrics(y_test, y_pred_price)
        m = {f"Test_{k}": v for k, v in m.items()}
        m.update(cross_validate_model(model, X_train, y_train))
        m["model"] = name
        results.append(m)
        fitted[name] = model
        print(f"  Test MAE={m['Test_MAE']:>12,.0f}  RMSE={m['Test_RMSE']:>12,.0f}"
              f"  R2={m['Test_R2']:.4f}  CV R2={m['CV_R2_mean']:.4f}"
              f"+-{m['CV_R2_std']:.4f}")

    comparison = (pd.DataFrame(results)
                  .set_index("model")
                  .sort_values("Test_RMSE"))
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (lower MAE/RMSE and higher R2 = better)")
    print("=" * 70)
    print(comparison.to_string(float_format=lambda v: f"{v:,.4f}"))

    # ------------------------------------------------------------------
    # 8. Select the best model by lowest test RMSE (documented criterion)
    # ------------------------------------------------------------------
    best_name = comparison.index[0]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)   # already rupee units
    print(f"\nSelected model: {best_name} (lowest test RMSE)")

    # ------------------------------------------------------------------
    # 9-12. Artifacts
    # ------------------------------------------------------------------
    metrics = {
        "selected_model": best_name,
        "selection_criterion": "lowest test RMSE (INR)",
        "random_state": config.RANDOM_STATE,
        "reference_year": config.REFERENCE_YEAR,
        "target_transform": config.TARGET_TRANSFORM,
        "test_size": config.TEST_SIZE,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "feature_list": list(X.columns),
        "target": target,
        "metrics_in": "original INR selling_price units",
        "models": results,
    }
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    comparison.to_csv(config.COMPARISON_CSV_PATH)

    joblib.dump(best_model, config.BEST_MODEL_PATH)
    with open(config.MODEL_META_PATH, "w") as fh:
        json.dump({
            "selected_model": best_name,
            "features": config.FEATURES,
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "target": target,
            "target_transform": config.TARGET_TRANSFORM,
            "random_state": config.RANDOM_STATE,
            "reference_year": config.REFERENCE_YEAR,
            "sklearn_pipeline": "preprocessor -> regressor, wrapped in "
                                "TransformedTargetRegressor(log1p/expm1)",
        }, fh, indent=2)

    plot_actual_vs_predicted(y_test, best_pred,
                             config.PLOTS_DIR / "actual_vs_predicted.png")
    plot_residuals(y_test, best_pred,
                   config.PLOTS_DIR / "residuals.png")
    plot_model_comparison(comparison,
                          config.PLOTS_DIR / "model_comparison.png")

    imp = feature_importance(best_model, X_test, y_test)
    imp.to_csv(config.ARTIFACTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(imp,
                            config.PLOTS_DIR / "feature_importance.png")

    errs = largest_errors(X_test, y_test, best_pred)
    errs.to_csv(config.ARTIFACTS_DIR / "largest_errors.csv", index=False)
    agg = plot_error_by_price_range(
        y_test, best_pred,
        config.PLOTS_DIR / "error_by_price_range.png")
    print("\nMean absolute error by actual price range:")
    print(agg.to_string(float_format=lambda v: f"{v:,.0f}"))
    print("\nLargest prediction errors:")
    print(errs.to_string(index=False))

    # ------------------------------------------------------------------
    # 13. Summary
    # ------------------------------------------------------------------
    best = comparison.iloc[0]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best model            : {best_name}")
    print(f"Test MAE              : {best['Test_MAE']:,.0f} INR")
    print(f"Test RMSE             : {best['Test_RMSE']:,.0f} INR")
    print(f"Test R2               : {best['Test_R2']:.4f}")
    print(f"CV R2 (train, 5-fold) : {best['CV_R2_mean']:.4f}"
          f" +/- {best['CV_R2_std']:.4f}")
    print(f"Artifacts             : {config.ARTIFACTS_DIR}")
    print(f"Saved model           : {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
