"""Assignment 9 CLI: run the full feature-importance workflow.

    python -m src.main          # train, evaluate, compute importances, save artifacts
    python -m src.main --skip-repro   # skip the 5-seed reproducibility check
"""

from __future__ import annotations

import argparse

from src import config
from src.data import FEATURES, load_and_prepare, make_splits
from src.importance import (
    compare_methods,
    group_to_original_features,
    impurity_importance,
    permutation_importance_df,
    plot_comparison,
    plot_top_importance,
    plot_top_permutation,
    save_csv,
)
from src.model import build_pipeline, regression_metrics, save_metrics

REPORT_TEMPLATE = """# Feature Importance Report — {dataset}

Generated automatically by `python -m src.main` (random_state={seed}).

## Dataset summary

- Source: {source_type}
- Origin: {origin}
- Rows used: {rows_final} (dropped {rows_dropped_missing_target} rows with missing target)
- Features: {n_features} ({n_num} numeric, {n_cat} categorical)
- Numeric: {numeric}
- Categorical: {categorical}

## Model metrics (test set, {test_size:.0%} split)

| Metric | Value |
|---|---:|
| MAE | {mae:,.0f} INR |
| RMSE | {rmse:,.0f} INR |
| R² | {r2:.4f} |

## Top 15 features — Random-Forest impurity importance

| Rank | Transformed feature | Importance |
|---:|---|---:|
{impurity_rows}

## Top 15 features — permutation importance (test set, {n_repeats} repeats)

Values are the MAE increase (INR) when the feature is shuffled; ± std across repeats.

| Rank | Transformed feature | MAE increase (INR) | ± std |
|---:|---|---:|---:|
{perm_rows}

## Grouped (original-feature) view

One-hot columns are aggregated back to their logical features.

| Original feature | Impurity share | Permutation (MAE, INR) | Transformed columns |
|---|---:|---:|---:|
{grouped_rows}

## Impurity vs permutation — where they disagree

{disagreement}

## Reproducibility check

Impurity importance is deterministic given the seed. Permutation importance
was recomputed with seeds {seed}–{seed_plus} (top-1 feature each time): {repro_line}
"""

DISAGREEMENT_TEMPLATE = """Normalised (0-1 within each method) importances of the top features, with
the absolute gap between the two methods. Larger gaps mark features whose
model usage (impurity) and measured usefulness (permutation) diverge —
typically features correlated with others (impurity splits credit across
them; permutation hides the redundancy because the model can compensate):

{table}
"""


def _fmt_impurity_rows(df) -> str:
    return "\n".join(
        f"| {i + 1} | {row.feature} | {row.importance:.4f} |"
        for i, row in enumerate(df.head(15).itertuples())
    )


def _fmt_perm_rows(df) -> str:
    return "\n".join(
        f"| {i + 1} | {row.feature} | {row.importance_mean:,.0f} | {row.importance_std:,.0f} |"
        for i, row in enumerate(df.head(15).itertuples())
    )


def _fmt_grouped_rows(df) -> str:
    return "\n".join(
        f"| {row.original_feature} | {row.impurity_importance:.4f} | "
        f"{row.permutation_importance:,.0f} | {row.n_transformed_columns} |"
        for row in df.head(15).itertuples()
    )


def _fmt_disagreement(comp) -> str:
    table = "\n".join(
        f"| {row.feature} | {row.impurity_norm:.3f} | {row.permutation_norm:.3f} | "
        f"{row.importance_gap:.3f} |"
        for row in comp.head(10).itertuples()
    )
    table = "| Feature | Impurity (norm) | Permutation (norm) | Gap |\n|---|---:|---:|---:|\n" + table
    return DISAGREEMENT_TEMPLATE.format(table=table)


def _reproducibility_check(X_train, X_test, y_train, y_test) -> str:
    """Recompute permutation importance under several seeds; report top-1 stability."""
    from sklearn.inspection import permutation_importance

    pipeline = build_pipeline().fit(X_train, y_train)
    top_features = []
    for seed in range(config.RANDOM_STATE, config.RANDOM_STATE + config.N_REPETITIONS):
        result = permutation_importance(
            pipeline, X_test, y_test,
            scoring=config.PERMUTATION_SCORING,
            n_repeats=5,
            random_state=seed,
            n_jobs=1,
        )
        top_features.append(X_test.columns[result.importances_mean.argmax()])
    stable = len(set(top_features)) == 1
    return (
        f"top-1 feature = {top_features[0]!r} in all {len(top_features)} runs "
        f"(stable: {stable})"
        if stable
        else f"varied across runs: {top_features}"
    )


def run(skip_repro: bool = False) -> dict:
    # 1-2. data
    X, y, provenance = load_and_prepare()
    X_train, X_test, y_train, y_test = make_splits(X, y)

    # 3-4. train + evaluate
    pipeline = build_pipeline().fit(X_train, y_train)
    metrics = regression_metrics(y_test, pipeline.predict(X_test))
    metrics["dataset"] = provenance
    metrics["random_state"] = config.RANDOM_STATE
    metrics["test_size"] = config.TEST_SIZE
    metrics["n_transformed_features"] = len(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    metrics["train_rows"] = int(len(X_train))
    metrics["test_rows"] = int(len(X_test))
    metrics["model"] = f"RandomForestRegressor(n_estimators={config.N_ESTIMATORS})"
    save_metrics(metrics)

    # 5-6. importances
    impurity_df = impurity_importance(pipeline)
    perm_df = permutation_importance_df(pipeline, X_test, y_test)
    grouped_df = group_to_original_features(impurity_df, perm_df)
    comp_df = compare_methods(impurity_df, perm_df)

    save_csv(impurity_df, config.FEATURE_IMPORTANCE_CSV)
    save_csv(perm_df, config.PERMUTATION_IMPORTANCE_CSV)
    save_csv(grouped_df, config.GROUPED_IMPORTANCE_CSV)

    # 7. plots
    plot_top_importance(
        impurity_df, "importance",
        f"Top {config.TOP_N} Random-Forest impurity importances",
        config.TOP_FEATURES_PNG, xlabel="impurity importance (share of variance reduction)",
    )
    plot_top_permutation(perm_df, config.TOP_PERMUTATION_PNG)
    plot_comparison(comp_df, config.COMPARISON_PNG)

    # 8. auto-generated report (actual values only)
    report = REPORT_TEMPLATE.format(
        dataset=provenance["dataset"],
        seed=config.RANDOM_STATE,
        source_type=provenance["source_type"],
        origin=provenance["origin"],
        rows_final=provenance["rows_final"],
        rows_dropped_missing_target=provenance["rows_dropped_missing_target"],
        n_features=provenance["n_features"],
        n_num=len(FEATURES) - len(provenance["categorical_features"]),
        n_cat=len(provenance["categorical_features"]),
        numeric=", ".join(provenance["numeric_features"]),
        categorical=", ".join(provenance["categorical_features"]),
        test_size=config.TEST_SIZE,
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        r2=metrics["R2"],
        n_repeats=config.PERMUTATION_N_REPEATS,
        impurity_rows=_fmt_impurity_rows(impurity_df),
        perm_rows=_fmt_perm_rows(perm_df),
        grouped_rows=_fmt_grouped_rows(grouped_df),
        disagreement=_fmt_disagreement(comp_df),
        seed_plus=config.RANDOM_STATE + config.N_REPETITIONS - 1,
        repro_line="skipped (run with repro check enabled)"
        if skip_repro
        else _reproducibility_check(X_train, X_test, y_train, y_test),
    )
    config.REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    config.REPORT_MD.write_text(report, encoding="utf-8")

    # 9. concise console summary
    print(f"Rows: {provenance['rows_final']}  |  transformed features: "
          f"{metrics['n_transformed_features']}")
    print(f"Model: {metrics['model']}")
    print(f"Test metrics: MAE={metrics['MAE']:,.0f} INR  RMSE={metrics['RMSE']:,.0f} INR  "
          f"R2={metrics['R2']:.4f}")
    print("\nTop 10 impurity features:")
    print(impurity_df.head(10).to_string(index=False))
    print("\nTop 10 permutation features (MAE increase, INR):")
    print(perm_df.head(10).to_string(index=False))
    print(f"\nArtifacts written to {config.ARTIFACTS_DIR} and {config.REPORT_MD}")
    return {"metrics": metrics, "impurity": impurity_df, "permutation": perm_df,
            "grouped": grouped_df, "comparison": comp_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-importance workflow (Assignment 9)")
    parser.add_argument("--skip-repro", action="store_true",
                        help="skip the multi-seed reproducibility check")
    args = parser.parse_args()
    run(skip_repro=args.skip_repro)


if __name__ == "__main__":
    main()
