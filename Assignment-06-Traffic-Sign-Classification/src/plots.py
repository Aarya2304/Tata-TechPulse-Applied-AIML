"""Matplotlib figures for the traffic-sign assignment."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import config

sns.set_theme(style="whitegrid")


def plot_class_distribution(counts: dict[int, int], out_path) -> None:
    ids = sorted(counts)
    values = [counts[i] for i in ids]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(ids, values, color="#4c9be8")
    mean = float(np.mean(values))
    ax.axhline(mean, color="#d95f02", ls="--", lw=1.2,
               label=f"mean = {mean:.0f}")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Training images")
    ax.set_title("GTSRB training-set class distribution (43 classes)")
    ax.set_xticks(range(0, 43, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle("Training history", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path) -> None:
    """43x43 matrix: raw counts annotated (small font) + per-row-normalised
    colour so rare classes remain readable."""
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(16, 13.5))
    sns.heatmap(cm_norm, ax=ax, cmap="viridis", cbar=True,
                square=False, linewidths=0.15, linecolor="#444444",
                annot=cm, fmt="d", annot_kws={"size": 5.5, "color": "white"})
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    ax.set_title("Confusion matrix - official GTSRB test set "
                 "(colour = row-normalised rate, numbers = counts)")
    ax.set_xticks(np.arange(0, 43, 2) + 0.5,
                  labels=[str(i) for i in range(0, 43, 2)])
    ax.set_yticks(np.arange(0, 43, 2) + 0.5,
                  labels=[str(i) for i in range(0, 43, 2)])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(test_df, y_true, y_pred, y_prob, out_path,
                       n: int = 12) -> None:
    """Grid of misclassified test images: true -> predicted + confidence."""
    from PIL import Image

    wrong = np.where(y_true != y_pred)[0][:n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 3.0 * rows))
    axes = np.atleast_2d(axes)
    class_names = {i: n for i, n in enumerate(config.CLASS_NAMES)}

    for ax in axes.ravel():
        ax.axis("off")
    for i, idx in enumerate(wrong):
        img = Image.open(test_df.iloc[idx]["image_path"])
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        conf = float(y_prob[idx][y_pred[idx]])
        ax.set_title(
            f"true {y_true[idx]} ({class_names[y_true[idx]][:18]})\n"
            f"pred {y_pred[idx]} ({class_names[y_pred[idx]][:18]}) "
            f"p={conf:.2f}",
            fontsize=8, color="#b00020")
    fig.suptitle("Misclassified test examples", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
