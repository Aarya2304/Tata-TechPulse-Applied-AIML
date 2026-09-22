"""Assignment 3 main entry point: full cleaning + preprocessing run.

Usage:
    python -m src.main
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.cleaning import (apply_domain_checks, count_capped_values,
                          detect_outliers_iqr, drop_exact_duplicates,
                          impute_dataframe, missing_before_after,
                          parse_features)
from src.load_data import load_raw
from src.preprocessing import (PreprocessingBuilder, leakage_verification,
                               scaling_comparison_stats)
from src.profiling import run_profiling

sns.set_theme(style="whitegrid")


def _boxplots(df_raw: pd.DataFrame, df_capped: pd.DataFrame,
              columns: list, out_path) -> None:
    """Side-by-side boxplots before/after outlier handling (shared y-scale)."""
    fig, axes = plt.subplots(
        len(columns), 2, figsize=(11, 3.1 * len(columns)),
        gridspec_kw={"width_ratios": [1, 1]})
    for i, col in enumerate(columns):
        # Shared x-scale across the pair for an honest comparison.
        lo = min(df_raw[col].min(), df_capped[col].min())
        hi = max(df_raw[col].max(), df_capped[col].max())
        pad = 0.05 * (hi - lo if hi > lo else 1)
        for j, (frame, label) in enumerate(
                [(df_raw, "Before (raw)"), (df_capped, "After (IQR-capped)")]):
            ax = axes[i, j] if len(columns) > 1 else axes[j]
            sns.boxplot(x=frame[col], ax=ax, color="#4c9be8")
            ax.set_title(f"{col} - {label}", fontsize=11)
            ax.set_xlabel("")
            ax.set_xlim(lo - pad, hi + pad)
    fig.suptitle("Outlier handling: before vs after IQR winsorization",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scaling_plots(full_df: pd.DataFrame, columns: list,
                   out_path) -> None:
    """Distributions before scaling + Standard vs Robust comparison."""
    from src.preprocessing import build_numeric_pipeline

    num_std = build_numeric_pipeline("standard")
    num_rob = build_numeric_pipeline("robust")
    raw = full_df[columns]
    std = pd.DataFrame(num_std.fit_transform(raw), columns=columns,
                       index=raw.index)
    rob = pd.DataFrame(num_rob.fit_transform(raw), columns=columns,
                       index=raw.index)

    fig, axes = plt.subplots(len(columns), 3,
                             figsize=(14, 2.9 * len(columns)))
    for i, col in enumerate(columns):
        sns.kdeplot(raw[col], ax=axes[i, 0], color="#444444", fill=True)
        axes[i, 0].set_title(f"{col} - raw", fontsize=10)
        sns.kdeplot(std[col], ax=axes[i, 1], color="#1f77b4", fill=True)
        axes[i, 1].set_title(f"{col} - StandardScaler", fontsize=10)
        sns.kdeplot(rob[col], ax=axes[i, 2], color="#d95f02", fill=True)
        axes[i, 2].set_title(f"{col} - RobustScaler", fontsize=10)
        for j in range(3):
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")
    fig.suptitle("Feature distributions: raw vs StandardScaler vs "
                 "RobustScaler", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    print("=" * 66)
    print("Assignment 3 - Data Cleaning & Preprocessing")
    print("=" * 66)

    # 1) load --------------------------------------------------------------
    raw = load_raw()
    print(f"\n[1] Loaded {raw.shape[0]} rows x {raw.shape[1]} cols")

    # 2) profile raw data ---------------------------------------------------
    profile = run_profiling(raw)
    print(f"[2] Profile saved -> {config.DATA_PROFILE_PATH.name} "
          f"({profile['total_missing_cells']} missing cells, "
          f"{profile['duplicate_rows']} duplicate rows)")

    # 3) type cleaning & feature parsing ------------------------------------
    parsed = parse_features(raw)
    print(f"[3] Parsed string numerics -> "
          f"{[dst for dst, _ in config.PARSED_NUM_COLUMNS.values()]} + brand")

    # 4) duplicates ----------------------------------------------------------
    deduped, n_dups = drop_exact_duplicates(parsed)
    print(f"[4] Removed {n_dups} exact duplicate rows "
          f"({len(deduped)} rows remain)")

    # 5) domain sanity checks -------------------------------------------------
    checked, domain_counts = apply_domain_checks(deduped)
    print(f"[5] Domain checks -> impossible values set to NaN: "
          f"{domain_counts}")

    # 6) outliers: detect on the cleaned data --------------------------------
    outlier_report, bounds = detect_outliers_iqr(
        checked, config.WINSORIZE_FEATURES + ["engine_cc", "torque_nm"])
    print("[6] IQR outliers per feature:")
    for _, row in outlier_report.iterrows():
        print(f"      {row['feature']:<16} {row['outlier_count']:>5} "
              f"({row['outlier_pct']:.2f}%)")

    # 7) missing-value handling (dataframe level, for the cleaned CSV) -------
    before_miss = checked.isna().sum().sum()
    imputed = impute_dataframe(
        checked,
        numeric_cols=config.NUMERIC_FEATURES,
        categorical_cols=config.CATEGORICAL_FEATURES,
    )
    after_miss = imputed.isna().sum().sum()
    miss_table = missing_before_after(checked, imputed)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    miss_table.to_csv(config.MISSING_REPORT_CSV_PATH, index=False)
    print(f"[7] Missing values: {before_miss} -> {after_miss}")

    # 8) outlier handling: winsorize the selected features --------------------
    from src.cleaning import IQRCapper

    capper = IQRCapper().fit(imputed[config.WINSORIZE_FEATURES])
    capped = capper.transform(imputed[config.WINSORIZE_FEATURES])
    capped_counts = count_capped_values(
        imputed[config.WINSORIZE_FEATURES], capped)
    capped_df = imputed.copy()
    for col in config.WINSORIZE_FEATURES:
        capped_df[col] = capped[col]
    n_capped = int(sum(capped_counts.values()))
    print(f"[8] Winsorized {n_capped} values across "
          f"{len(config.WINSORIZE_FEATURES)} features")

    outlier_report.to_csv(config.OUTLIER_REPORT_CSV_PATH, index=False)

    # 9) scaling comparison (Standard vs Robust) ------------------------------
    scaler_stats = scaling_comparison_stats(
        capped_df, config.SCALER_COMPARISON_FEATURES)
    pd.DataFrame(scaler_stats).T.to_csv(
        config.REPORTS_DIR / "scaling_comparison.csv")
    print(f"[9] Scaler comparison computed for "
          f"{len(config.SCALER_COMPARISON_FEATURES)} features")

    # 10) leakage verification (train-only fitting) ---------------------------
    rng = pd.Series(
        __import__("numpy").random.RandomState(config.RANDOM_STATE)
        .rand(len(capped_df)))
    test_mask = rng < config.TEST_SIZE
    train_df = capped_df[~test_mask]
    test_df = capped_df[test_mask]
    leakage = leakage_verification(train_df, test_df)
    config.LEAKAGE_REPORT_PATH.write_text(json.dumps(leakage, indent=2))
    print(f"[10] Leakage check -> imputer==train: "
          f"{leakage['imputer_medians_equal_train_medians']}, "
          f"scaler==capped-train: "
          f"{leakage['scaler_means_equal_capped_train_means']}, "
          f"unchanged after test transform: "
          f"{leakage['scaler_stats_unchanged_after_test_transform']}")

    # 11) fit the full pipeline & save processed matrix ------------------------
    builder = PreprocessingBuilder(scaler="robust")
    X_full = builder.build().fit_transform(
        capped_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES])
    feature_names = builder.feature_names
    processed = pd.DataFrame(X_full, columns=feature_names)
    processed.insert(0, "selling_price_raw", capped_df[config.TARGET].values)
    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"[11] Processed matrix: {processed.shape} -> "
          f"{config.PROCESSED_DATA_PATH.name}")

    # 12) plots ----------------------------------------------------------------
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _boxplots(imputed[config.WINSORIZE_FEATURES], capped_df[
        config.WINSORIZE_FEATURES], config.WINSORIZE_FEATURES,
        config.PLOTS_DIR / "outliers_before_after.png")
    _scaling_plots(capped_df, config.SCALER_COMPARISON_FEATURES,
                   config.PLOTS_DIR / "scaling_comparison.png")
    # Missing-after plot
    from src.profiling import save_missing_plot
    save_missing_plot(processed, config.PLOTS_DIR /
                      "missing_values_after.png")

    # 13) cleaning report -------------------------------------------------------
    transformations = [
        "Parsed 'mileage' strings ('23.4 kmpl' / '17.3 km/kg') to numeric "
        "mileage_kmpl",
        "Parsed 'engine' ('1248 CC') to engine_cc",
        "Parsed 'max_power' ('74 bhp') to max_power_bhp",
        "Parsed 'torque' strings (Nm / kgm formats, kgm*9.80665) to "
        "torque_nm; dropped raw torque column",
        "Derived 'brand' from the first word of 'name'; dropped raw 'name'",
        f"Removed {n_dups} exact duplicate rows",
        "Domain checks: km_driven < "
        f"{config.MIN_PLAUSIBLE_KM_DRIVEN:.0f}, mileage_kmpl == 0 and "
        "max_power_bhp == 0 converted to NaN then imputed",
        "Median imputation for numeric features",
        "Most-frequent imputation for categorical features",
        f"IQR (k={config.IQR_K}) winsorization applied to "
        f"{config.WINSORIZE_FEATURES}",
        "StandardScaler and RobustScaler compared on numeric features",
        "OneHotEncoder(handle_unknown='ignore') for categorical features",
    ]
    cleaning_report = {
        "dataset": "CarDekho 'Car details v3' (Kaggle: nehalbirla/"
                   "vehicle-dataset-from-cardekho)",
        "original_row_count": int(raw.shape[0]),
        "original_column_count": int(raw.shape[1]),
        "duplicate_rows_removed": int(n_dups),
        "rows_after_dedup": int(len(deduped)),
        "rows_removed_other": 0,
        "columns_removed": ["torque (raw string, replaced by torque_nm)",
                            "name (replaced by derived brand)"],
        "final_row_count": int(len(capped_df)),
        "missing_values_before_total": int(before_miss),
        "missing_values_after_total": int(after_miss),
        "missing_before_by_column": {
            k: int(v) for k, v in raw.isna().sum().items() if v > 0},
        "domain_anomalies_fixed": domain_counts,
        "outliers_detected": {
            row["feature"]: int(row["outlier_count"])
            for _, row in outlier_report.iterrows()},
        "outliers_capped": {k: int(v) for k, v in capped_counts.items()},
        "outlier_strategy": ("1.5*IQR winsorization (capping) for "
                             "km_driven, selling_price, mileage_kmpl, "
                             "max_power_bhp; engine_cc, torque_nm, year, "
                             "seats retained uncapped (legitimate variety)"),
        "imputation": {"numeric": "median", "categorical": "most_frequent"},
        "scaling_comparison": scaler_stats,
        "leakage_verification": leakage,
        "transformations_applied": transformations,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "processed_matrix_shape": list(processed.shape),
    }
    config.CLEANING_REPORT_PATH.write_text(
        json.dumps(cleaning_report, indent=2), encoding="utf-8")
    print(f"[13] Cleaning report -> {config.CLEANING_REPORT_PATH.name}")

    # 14) concise console summary ------------------------------------------------
    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(f"rows            : {raw.shape[0]} -> {len(capped_df)} "
          f"(-{n_dups} duplicates)")
    print(f"missing cells   : {before_miss} -> {after_miss}")
    print(f"outliers capped : {n_capped}")
    print(f"processed shape : {processed.shape}")
    print(f"leakage-safe    : {leakage['imputer_medians_equal_train_medians']}"
          f" / {leakage['scaler_means_equal_capped_train_means']} / "
          f"{leakage['scaler_stats_unchanged_after_test_transform']}")
    print(f"artifacts       : {config.ARTIFACTS_DIR}")
    return cleaning_report


if __name__ == "__main__":
    main()
