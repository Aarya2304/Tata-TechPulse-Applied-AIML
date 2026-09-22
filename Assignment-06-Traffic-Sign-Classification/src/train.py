"""End-to-end GTSRB CNN training. Run with:  python -m src.train"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.data_loader import (class_distribution, discover_training_data,
                             ensure_downloaded, load_test_dataframe,
                             stratified_split, validate_classes)
from src.model import build_cnn, compile_model, make_callbacks
from src.preprocessing import make_dataset, make_test_dataset


def set_seeds(seed: int = config.RANDOM_STATE) -> None:
    """Best-effort determinism: Python, NumPy and TensorFlow seeds.

    Note: full bit-for-bit reproducibility is NOT guaranteed (GPU kernels
    and oneDNN reduction order are non-deterministic); seeds make runs
    comparable but not identical.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds()

    print("=" * 70)
    print("TRAFFIC SIGN CLASSIFICATION (GTSRB) - CNN TRAINING")
    print("TensorFlow:", tf.__version__, "| GPUs:",
          tf.config.list_physical_devices("GPU") or "none (CPU)")
    print("=" * 70)

    # 1. Data ---------------------------------------------------------------
    ensure_downloaded()
    paths, labels, counts = discover_training_data()
    validate_classes(labels)
    print(f"\nTraining pool: {len(paths)} images, {len(counts)} classes")
    print(f"Class sizes: min={min(counts.values())} "
          f"max={max(counts.values())} (imbalance "
          f"{max(counts.values()) / min(counts.values()):.1f}x)")

    test_df = load_test_dataframe()
    print(f"Official test set: {len(test_df)} images")

    train_paths, train_labels, val_paths, val_labels = stratified_split(
        paths, labels)
    print(f"Stratified split: {len(train_paths)} train / "
          f"{len(val_paths)} validation "
          f"(fraction={config.VALIDATION_FRACTION}, seed={config.RANDOM_STATE})")

    # 2. tf.data pipelines ----------------------------------------------------
    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels, training=False)
    test_ds = make_test_dataset(list(test_df["image_path"]))

    # 3. Class weights from TRAINING data only --------------------------------
    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes,
                                   y=np.asarray(train_labels))
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Class weights computed from training data only "
          f"({len(class_weight)} classes)")

    # 4. Model -----------------------------------------------------------------
    model = compile_model(build_cnn())
    model.summary()

    # 5. Train -------------------------------------------------------------------
    callbacks = make_callbacks(config.MODEL_PATH)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    # `restore_best_weights` already left the best checkpoint in memory;
    # the checkpoint on disk is the val_accuracy-max epoch.
    print(f"\nBest epoch (val_accuracy): "
          f"{int(np.argmax(history.history['val_accuracy'])) + 1} / "
          f"{len(history.history['val_accuracy'])}")

    # 6. Evaluate on the official test set -------------------------------------
    print("\nEvaluating on the official GTSRB test set...")
    y_true = test_df["class_id"].to_numpy()
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    from sklearn.metrics import accuracy_score
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        zero_division=0)
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.average(prec, weights=np.bincount(
            y_true, minlength=config.NUM_CLASSES))),
        "weighted_recall": float(np.average(rec, weights=np.bincount(
            y_true, minlength=config.NUM_CLASSES))),
        "weighted_f1": float(np.average(f1, weights=np.bincount(
            y_true, minlength=config.NUM_CLASSES))),
        "n_test": int(len(y_true)),
        "n_incorrect": int((y_true != y_pred).sum()),
        "best_epoch": int(np.argmax(history.history["val_accuracy"])) + 1,
        "epochs_run": len(history.history["val_accuracy"]),
        "final_val_accuracy": float(max(history.history["val_accuracy"])),
        "seed": config.RANDOM_STATE,
        "note": "seeds set for python/numpy/tensorflow; bit-for-bit "
                "reproducibility is not guaranteed on this backend",
    }

    # classification report CSV
    report = classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_NAMES, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(config.REPORT_CSV_PATH)

    with open(config.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(config.CLASS_NAMES_PATH, "w") as fh:
        json.dump({str(i): n for i, n in enumerate(config.CLASS_NAMES)},
                  fh, indent=2)

    # 7. Plots -----------------------------------------------------------------
    from src.plots import (plot_class_distribution, plot_confusion_matrix,
                           plot_history, plot_misclassified)
    plot_class_distribution(counts,
                            config.PLOTS_DIR / "class_distribution.png")
    plot_history(history.history, config.PLOTS_DIR / "training_history.png")
    plot_confusion_matrix(y_true, y_pred,
                          config.PLOTS_DIR / "confusion_matrix.png")
    plot_misclassified(test_df, y_true, y_pred, y_prob,
                       config.PLOTS_DIR / "misclassified_examples.png")

    # 8. Error analysis summary --------------------------------------------------
    errors = y_true != y_pred
    print(f"\nIncorrect predictions: {metrics['n_incorrect']} / {len(y_true)}")
    if errors.any():
        pair_counts = (pd.DataFrame({"true": y_true[errors],
                                     "pred": y_pred[errors]})
                       .groupby(["true", "pred"]).size()
                       .sort_values(ascending=False))
        print("Top confused class pairs (true -> pred: count):")
        for (t, p), c in pair_counts.head(8).items():
            print(f"  {t:2d} ({config.CLASS_NAMES[t][:28]:<28}) -> "
                  f"{p:2d} ({config.CLASS_NAMES[p][:28]:<28}): {c}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test accuracy        : {metrics['test_accuracy']:.4f}")
    print(f"Macro P / R / F1     : {metrics['macro_precision']:.4f} / "
          f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"Weighted P / R / F1  : {metrics['weighted_precision']:.4f} / "
          f"{metrics['weighted_recall']:.4f} / {metrics['weighted_f1']:.4f}")
    print(f"Incorrect            : {metrics['n_incorrect']} / {len(y_true)}")
    print(f"Model saved          : {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
