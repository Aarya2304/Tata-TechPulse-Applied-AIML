"""End-to-end training pipeline for the LSTM sentiment classifier.

Run:  python -m src.train

Steps: set seeds → load real dataset → clean/inspect → stratified 80/10/10
split → fit tokenizer on train only → pad sequences → build LSTM → train
(class-weighted, EarlyStopping/ReduceLR/ModelCheckpoint) → evaluate on the
untouched test set → save metrics/plots/sample predictions → error analysis.
"""

from __future__ import annotations

import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.data_loader import clean_dataframe, dataset_report, download_dataset, load_raw_dataframe
from src.model import build_lstm_model, make_callbacks
from src.preprocessing import (
    build_sequences,
    clean_text,
    compute_class_weights,
    fit_tokenizer,
    make_splits,
    save_tokenizer,
)

CLASS_NAMES = config.CLASS_NAMES


def set_seeds(seed: int = config.RANDOM_STATE) -> None:
    """Seed python/numpy/tensorflow for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def save_metrics(metrics: dict, path=None) -> None:
    path = path or config.METRICS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_metadata(metadata: dict, path=None) -> None:
    path = path or config.METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_class_distribution(y: np.ndarray, out_path=None) -> None:
    out_path = out_path or config.PLOTS_DIR / "class_distribution.png"
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=config.PLOT_DPI)
    ax.bar([CLASS_NAMES[int(c)] for c in counts.index], counts.values,
           color=["#d95f02", "#1b9e77"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Sentiment class distribution (cleaned dataset)")
    ax.set_ylabel("reviews")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_text_length_distribution(texts: list[str], out_path=None) -> None:
    out_path = out_path or config.PLOTS_DIR / "text_length_distribution.png"
    lengths = pd.Series([len(t.split()) for t in texts])
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=config.PLOT_DPI)
    ax.hist(lengths.clip(upper=400), bins=40, color="#4575b4")
    ax.axvline(config.SEQUENCE_LENGTH, color="crimson", linestyle="--",
               label=f"SEQUENCE_LENGTH = {config.SEQUENCE_LENGTH}")
    ax.set_xlabel("review length (words, clipped at 400)")
    ax.set_ylabel("reviews")
    ax.set_title("Review text length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_history(history, out_path=None) -> None:
    out_path = out_path or config.PLOTS_DIR / "training_history.png"
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=config.PLOT_DPI)
    axes[0].plot(epochs, hist["accuracy"], label="train")
    axes[0].plot(epochs, hist["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, hist["loss"], label="train")
    axes[1].plot(epochs, hist["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle("LSTM training history")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path=None) -> None:
    out_path = out_path or config.PLOTS_DIR / "confusion_matrix.png"
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=config.PLOT_DPI)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        ax=ax,
    )
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sample_predictions(samples: pd.DataFrame, out_path=None) -> None:
    """Bar chart of prediction probabilities for the saved sample reviews."""
    out_path = out_path or config.PLOTS_DIR / "sample_predictions.png"
    fig, ax = plt.subplots(figsize=(9, 6), dpi=config.PLOT_DPI)
    y = np.arange(len(samples))[::-1]
    colors = ["#1b9e77" if row.correct else "#d95f02" for _, row in samples.iterrows()]
    ax.barh(y, samples.probability, color=colors)
    ax.set_yticks(y)
    labels = [
        f"[{'OK' if r.correct else 'ERR'}] {str(r.text)[:52]}..."
        for r in samples.itertuples()
    ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(positive)")
    ax.set_title("Sample test predictions (green = correct, orange = wrong)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    set_seeds()
    for directory in (config.MODELS_DIR, config.PLOTS_DIR, config.REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1-3. load + clean + inspect the real dataset
    download_dataset()
    raw_df = load_raw_dataframe()
    df, clean_stats = clean_dataframe(raw_df)
    print(dataset_report(df, clean_stats))

    # 4. stratified 80/10/10 split (test stays untouched until evaluation)
    splits = make_splits(df)
    print(f"split sizes: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    # 5-6. tokenizer fitted on TRAIN text only, then pad every split
    train_texts = [clean_text(t) for t in splits.train[config.TEXT_COLUMN]]
    val_texts = [clean_text(t) for t in splits.val[config.TEXT_COLUMN]]
    test_texts = [clean_text(t) for t in splits.test[config.TEXT_COLUMN]]

    tokenizer = fit_tokenizer(train_texts)
    seq = build_sequences(
        tokenizer, train_texts, val_texts, test_texts,
        splits.train[config.LABEL_COLUMN].to_numpy(),
        splits.val[config.LABEL_COLUMN].to_numpy(),
        splits.test[config.LABEL_COLUMN].to_numpy(),
    )
    print(f"tokenizer vocabulary: {len(tokenizer.word_index)} words fitted, "
          f"using {seq.vocab_size} embedding rows, sequence length {seq.sequence_length}")

    # 7. class weights from TRAIN only (documented imbalance handling)
    class_weights = compute_class_weights(seq.y_train)
    print(f"class weights (train): {class_weights}")

    # 8. build + train the LSTM
    model = build_lstm_model(vocab_size=seq.vocab_size, sequence_length=seq.sequence_length)
    model.summary()
    history = model.fit(
        seq.X_train, seq.y_train,
        validation_data=(seq.X_val, seq.y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(config.TRAINING_LOG_CSV, index=False)

    # 9-10. best model + tokenizer + metadata persisted
    save_tokenizer(tokenizer)
    metadata = {
        "model_path": str(config.MODEL_PATH.name),
        "architecture": "Embedding -> LSTM -> Dropout -> Dense(relu) -> Dense(sigmoid)",
        "vocab_size": int(seq.vocab_size),
        "fitted_vocabulary": int(len(tokenizer.word_index)),
        "sequence_length": int(seq.sequence_length),
        "embedding_dim": config.EMBEDDING_DIM,
        "lstm_units": config.LSTM_UNITS,
        "dense_units": config.DENSE_UNITS,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "max_epochs": config.MAX_EPOCHS,
        "epochs_run": len(history.history["loss"]),
        "random_state": config.RANDOM_STATE,
        "class_mapping": {str(k): v for k, v in config.CLASS_NAMES.items()},
        "class_weights": {str(k): round(v, 4) for k, v in class_weights.items()},
        "split_sizes": seq.split_sizes,
        "dataset": config.DATASET_NAME,
        "dataset_url": config.DATASET_URL,
        "cleaning_stats": clean_stats,
        "threshold": 0.5,
    }
    save_metadata(metadata)

    # 11-12. evaluate on the untouched test set
    y_prob = model.predict(seq.X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = seq.y_test.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # binary P/R/F1 are reported for the POSITIVE class (sklearn default);
        # the minority (negative) class metrics follow from the confusion
        # matrix and the classification_report CSV.
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "test_f1": round(float(f1), 4),
        "test_roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "test_pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "val_accuracy_best": round(float(max(history.history["val_accuracy"])), 4),
        "threshold": 0.5,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "split_sizes": seq.split_sizes,
        "epochs_run": len(history.history["loss"]),
    }
    save_metrics(metrics)
    print(json.dumps(metrics, indent=2))

    # 13. classification report CSV (per-class precision/recall/F1)
    report = classification_report(
        y_true, y_pred, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(config.CLASSIFICATION_REPORT_CSV)

    # 14-16. plots
    plot_class_distribution(seq.y_train)
    plot_text_length_distribution(df[config.TEXT_COLUMN].tolist())
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred)

    # 15. sample predictions: mix of correct and incorrect test reviews
    test_df = splits.test.copy()
    test_df["probability"] = y_prob
    test_df["predicted"] = y_pred
    test_df["correct"] = test_df["predicted"] == test_df[config.LABEL_COLUMN]
    wrong = test_df[~test_df.correct]
    right = test_df[test_df.correct]
    n_wrong = min(len(wrong), config.MAX_SAMPLE_PREDICTIONS // 2)
    n_right = min(len(right), config.MAX_SAMPLE_PREDICTIONS - n_wrong)
    samples = pd.concat(
        [wrong.sample(n=n_wrong, random_state=config.RANDOM_STATE),
         right.sample(n=n_right, random_state=config.RANDOM_STATE)]
    ).sample(frac=1.0, random_state=config.RANDOM_STATE)
    samples_out = samples.assign(
        text=samples[config.TEXT_COLUMN].str.slice(0, 180),
        true_sentiment=samples[config.LABEL_COLUMN].map(CLASS_NAMES),
        predicted_sentiment=samples["predicted"].map(CLASS_NAMES),
    )[["text", config.RATING_COLUMN, "true_sentiment", "predicted_sentiment",
       "probability", "correct"]]
    samples_out.to_csv(config.SAMPLE_PREDICTIONS_CSV, index=False)
    plot_sample_predictions(samples_out)

    # 17. error analysis (measured, not invented)
    errors = test_df[~test_df.correct]
    print("\n--- error analysis (test set) ---")
    print(f"misclassified: {len(errors)} / {len(test_df)}")
    if len(errors):
        err_len = errors[config.TEXT_COLUMN].str.split().str.len()
        ok_len = test_df[test_df.correct][config.TEXT_COLUMN].str.split().str.len()
        print(f"mean words, errors vs correct: {err_len.mean():.0f} vs {ok_len.mean():.0f}")
        print(f"errors with prob in [0.4, 0.6] (uncertain): "
              f"{int(errors.probability.between(0.4, 0.6).sum())} / {len(errors)}")
        neg = errors[errors[config.LABEL_COLUMN] == 0]
        if len(neg):
            print(f"false-alarm rate on true negatives: {len(neg)} / "
                  f"{int((test_df[config.LABEL_COLUMN] == 0).sum())} true negatives")
    print("\nArtifacts written:")
    for p in (config.METRICS_JSON, config.CLASSIFICATION_REPORT_CSV,
              config.SAMPLE_PREDICTIONS_CSV, config.MODEL_PATH,
              config.TOKENIZER_PATH, config.METADATA_PATH):
        print(f"  {p}  ({p.stat().st_size/1024:.0f} KB)" if p.exists() else f"  MISSING: {p}")


if __name__ == "__main__":
    main()
