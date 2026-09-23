"""Feature-importance computation: impurity-based and permutation-based.

Both methods operate on the *transformed* feature space (numeric features
plus one-hot columns produced by the ColumnTransformer); one-hot columns are
then also aggregated back to their original logical feature groups for a
second, more interpretable view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def transformed_feature_names(pipeline) -> list[str]:
    """Feature names of the model input after the ColumnTransformer.

    Uses ``get_feature_names_out`` with ``verbose_feature_names_out=False``,
    so names are the raw column names for numeric features and
    ``column_value`` for one-hot columns (e.g. ``brand_Maruti``).
    """
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


# ---------------------------------------------------------------------------
# 1. Impurity (model) importance
# ---------------------------------------------------------------------------


def impurity_importance(pipeline) -> pd.DataFrame:
    """Random-Forest impurity importances per transformed feature.

    impurity importance = total reduction of the splitting criterion
    (variance for regression) attributed to each feature, averaged over all
    trees and normalised to sum to 1 by sklearn.
    """
    names = transformed_feature_names(pipeline)
    values = pipeline.named_steps["model"].feature_importances_
    df = pd.DataFrame({"feature": names, "importance": np.asarray(values, dtype=float)})
    df["method"] = "impurity"
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance_df(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series,
    n_repeats: int = config.PERMUTATION_N_REPEATS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance on the untouched test set (transformed space).

    The test data is first passed through the fitted ColumnTransformer, and
    permutations are applied to the **transformed columns** (numeric features
    plus individual one-hot dummies) so the reported names correspond exactly
    to the impurity-importance table. Caveat: permuting one dummy of a
    multi-level categorical creates value combinations unseen in training;
    the grouped view below mitigates this by aggregating dummies back to
    their logical feature.

    Scoring: ``neg_mean_absolute_error`` — each value is the *decrease in
    negative MAE* when one feature's values are randomly shuffled, i.e. the
    MAE increase in rupees caused by destroying that feature's information.
    Values are ≥ 0 when a feature is useful; ~0 means the model barely
    relies on it, and negative values mean shuffling actually *improved* MAE
    (noise). Variability across the repeats is reported as the standard
    deviation.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_trans = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=transformed_feature_names(pipeline),
        index=X_test.index,
    )
    result = permutation_importance(
        model, X_trans.to_numpy(), y_test,  # ndarray: model was fitted on arrays
        scoring=config.PERMUTATION_SCORING,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": list(X_trans.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    df["method"] = "permutation"
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grouping one-hot columns back to logical features
# ---------------------------------------------------------------------------


def group_to_original_features(impurity_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate both importance tables to the original logical features.

    One-hot columns share the importance of their parent feature. Grouping is
    done by exact prefix matching against the known feature names, so a brand
    like ``brand_Isuzu`` maps to ``brand`` and numeric columns stay unchanged.
    Returns one row per original feature with the two methods side by side
    plus the group size (number of transformed columns behind it).
    """
    def owner_of(column: str) -> str:
        for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if column == feature or column.startswith(feature + "_"):
                return feature
        return "other"

    imp = impurity_df.copy()
    imp["group"] = imp["feature"].map(owner_of)
    imp_grouped = imp.groupby("group")["importance"].sum()

    perm = perm_df.copy()
    perm["group"] = perm["feature"].map(owner_of)
    perm_grouped = perm.groupby("group")["importance_mean"].sum()
    perm_var = (perm.groupby("group")["importance_std"].var()).fillna(0.0)

    grouped = pd.DataFrame(
        {
            "impurity_importance": imp_grouped,
            "permutation_importance": perm_grouped,
            "permutation_std_across_groups": np.sqrt(perm_var),
        }
    )
    grouped["n_transformed_columns"] = imp.groupby("group")["feature"].count()
    grouped["permutation_units"] = "MAE increase (INR)"
    grouped["impurity_units"] = "normalised (sums to 1 over all groups)"
    return grouped.sort_values("impurity_importance", ascending=False).reset_index() \
        .rename(columns={"group": "original_feature"})


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_methods(impurity_df: pd.DataFrame, perm_df: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    """Side-by-side normalised comparison of the two methods.

    Both columns are min-max normalised to [0, 1] *within each method* so a
    raw impurity share (unitless) can be compared with a raw MAE-increase
    (rupees) without pretending they share a unit.
    """
    top_n = top_n or config.TOP_N
    imp = impurity_df.set_index("feature")["importance"]
    perm = perm_df.set_index("feature")["importance_mean"]
    common = imp.index.intersection(perm.index)

    def normalise(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo) if hi > lo else series * 0.0

    comp = pd.DataFrame(
        {
            "impurity": imp.loc[common],
            "permutation": perm.loc[common],
            "impurity_norm": normalise(imp.loc[common]),
            "permutation_norm": normalise(perm.loc[common]),
        }
    )
    comp["importance_gap"] = (comp["impurity_norm"] - comp["permutation_norm"]).abs()
    return comp.sort_values("impurity_norm", ascending=False).head(top_n).reset_index() \
        .rename(columns={"index": "feature"})


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_top_importance(df: pd.DataFrame, value_col: str, title: str, out_path,
                        top_n: int = None, xlabel: str = "importance") -> None:
    """Horizontal bar chart of the top-N features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_n = top_n or config.TOP_N
    top = df.head(top_n).iloc[::-1]  # largest at the top of the chart
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.32 * len(top))), dpi=config.PLOT_DPI)
    ax.barh(top["feature"], top[value_col], color=config.TOP_N_BAR_COLORS)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for y, v in enumerate(top[value_col]):
        ax.text(v, y, f" {v:.3f}" if v < 10 else f" {v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_permutation(df: pd.DataFrame, out_path, top_n: int = None) -> None:
    """Horizontal bar chart with error bars from the repeat variability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_n = top_n or config.TOP_N
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.32 * len(top))), dpi=config.PLOT_DPI)
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="#de6f57",
            error_kw={"lw": 1, "capsize": 2})
    ax.set_xlabel("MAE increase when the feature is shuffled (INR)")
    ax.set_title(f"Top {len(top)} permutation importances (test set, "
                 f"{config.PERMUTATION_N_REPEATS} repeats)")
    for y, v in enumerate(top["importance_mean"]):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_comparison(comp_df: pd.DataFrame, out_path, top_n: int = None) -> None:
    """Grouped bars: normalised impurity vs normalised permutation importance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_n = top_n or min(15, len(comp_df))
    top = comp_df.head(top_n)
    y = np.arange(len(top))
    height = 0.4
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.4 * len(top))), dpi=config.PLOT_DPI)
    ax.barh(y - height / 2, top["impurity_norm"], height,
            label="Random-Forest impurity (normalised)", color="#2c7fb8")
    ax.barh(y + height / 2, top["permutation_norm"], height,
            label="Permutation (normalised)", color="#de6f57")
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("normalised importance (0-1 within each method)")
    ax.set_title("Impurity vs permutation importance — top common features")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
