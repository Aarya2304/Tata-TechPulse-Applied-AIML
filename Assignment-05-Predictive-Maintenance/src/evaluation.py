"""Evaluation: imbalance-aware metrics and matplotlib figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from src import config

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_prob, threshold=config.DECISION_THRESHOLD) -> dict:
    """All required metrics for the FAILURE class (positive label 1).

    ``y_prob`` is the predicted probability of failure; the class decision
    uses ``threshold`` (0.5 for all models, so comparisons are fair).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "true_positives": int(np.sum((y_true == 1) & (y_pred == 1))),
        "false_positives": int(np.sum((y_true == 0) & (y_pred == 1))),
        "false_negatives": int(np.sum((y_true == 1) & (y_pred == 0))),
        "true_negatives": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def select_best(comparison: pd.DataFrame) -> str:
    """Apply the predefined criterion: best failure F1, then PR-AUC, then
    recall. (The criterion was fixed in config.py before training.)"""
    ranked = comparison.sort_values(
        by=[config.SELECTION_METRIC] + config.SELECTION_TIEBREAKERS,
        ascending=False)
    return str(ranked.index[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_class_distribution(y: pd.Series, out_path) -> None:
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values,
                  color=["#4c9be8", "#d95f02"])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count}\n({count / counts.sum() * 100:.2f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of samples")
    ax.set_title("Target distribution - Machine failure (10,000 samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_prob, out_path,
                          threshold=config.DECISION_THRESHOLD) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"])
    ax.set_title(f"Confusion matrix (threshold={threshold})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true, prob_dict: dict, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, y_prob in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (failure class = positive)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(y_true, prob_dict: dict, out_path) -> None:
    prevalence = float(np.mean(np.asarray(y_true) == 1))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, y_prob in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(prevalence, color="k", ls="--", lw=1,
               label=f"Chance ({prevalence:.3f})")
    ax.set_xlabel("Recall (failure class)")
    ax.set_ylabel("Precision (failure class)")
    ax.set_title("Precision-Recall curves (failure class = positive)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison: pd.DataFrame, out_path) -> None:
    metrics = [("accuracy", "Accuracy"),
               ("precision", "Precision (failure)"),
               ("recall", "Recall (failure)"),
               ("f1", "F1 (failure)"),
               ("roc_auc", "ROC-AUC"),
               ("average_precision", "PR-AUC / AP")]
    fig, axes = plt.subplots(1, 6, figsize=(21, 4.6))
    df = comparison.reset_index()
    for ax, (metric, title) in zip(axes, metrics):
        bars = ax.bar(df["model"], df[metric], color="#4c9be8")
        bars[int(df[metric].idxmax())].set_color("#d95f02")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        for bar, value in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .01,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model comparison (orange = best per metric)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def error_analysis(X_test: pd.DataFrame, y_true, y_prob,
                   threshold=config.DECISION_THRESHOLD,
                   n_examples: int = 8) -> dict:
    """Summarise false positives / false negatives with example rows."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    cols = [c for c in config.FEATURES if c in X_test.columns]
    fp = X_test.iloc[fp_idx][cols].copy()
    fp["failure_probability"] = np.asarray(y_prob)[fp_idx]
    fn = X_test.iloc[fn_idx][cols].copy()
    fn["failure_probability"] = np.asarray(y_prob)[fn_idx]

    summary = {
        "n_false_positives": int(len(fp_idx)),
        "n_false_negatives": int(len(fn_idx)),
        "failure_recall": float(1 - len(fn_idx) / max(np.sum(y_true == 1), 1)),
        "false_negatives": fn.head(n_examples).reset_index(drop=True),
        "false_positives": fp.head(n_examples).reset_index(drop=True),
        "fn_probability_stats": {
            "mean": float(np.mean(np.asarray(y_prob)[fn_idx])) if len(fn_idx) else None,
            "max": float(np.max(np.asarray(y_prob)[fn_idx])) if len(fn_idx) else None,
        },
    }
    return summary
