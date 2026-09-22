"""Prediction CLI: classify a traffic-sign image with the saved CNN.

Usage:
    python -m src.predict path/to/sign.png
    python -m src.predict                # demo mode (first test images)

The saved model contains the full preprocessing contract (32x32 RGB,
pixel/255 normalisation happens here exactly as in training).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src import config
from src.preprocessing import preprocess_path


def load_class_names() -> dict[int, str]:
    """Load the class-id -> name mapping saved at training time."""
    with open(config.CLASS_NAMES_PATH) as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def preprocess_image(image_path: str):
    """File path -> float32 [1,H,W,3] in [0,1] (same path as training)."""
    tensor = preprocess_path(np.asarray(image_path))
    return tf_expand(tensor)


def tf_expand(img):
    import tensorflow as tf
    return tf.expand_dims(img, 0)


def predict_image(model, image_path: str, class_names: dict[int, str]) -> dict:
    prob = model.predict(tf_expand(preprocess_path(np.asarray(image_path))),
                         verbose=0)[0]
    class_id = int(np.argmax(prob))
    return {"class_id": class_id,
            "class_name": class_names[class_id],
            "confidence": float(prob[class_id]),
            "probabilities": prob}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a traffic-sign image with the saved CNN.")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to a traffic-sign image (ppm/png/jpg).")
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {config.MODEL_PATH}.\n"
                         "Run 'python -m src.train' first.")

    import tensorflow as tf
    model = tf.keras.models.load_model(config.MODEL_PATH)
    class_names = load_class_names()
    print(f"Loaded model from {config.MODEL_PATH}")

    if args.image:
        result = predict_image(model, args.image, class_names)
        print(f"\nImage   : {args.image}")
        print(f"Predicted class ID  : {result['class_id']}")
        print(f"Predicted class name: {result['class_name']}")
        print(f"Confidence          : {result['confidence']:.4f}")
        return

    # Demo mode: run the model on a few official test images.
    from src.data_loader import load_test_dataframe
    test_df = load_test_dataframe().head(5)
    print("\nDemo predictions on official GTSRB test images:\n")
    for _, row in test_df.iterrows():
        r = predict_image(model, row["image_path"], class_names)
        correct = "OK " if r["class_id"] == row["class_id"] else "MISS"
        print(f"  [{correct}] {row['image_path'].split('/')[-1]}  "
              f"true={row['class_id']:2d} ({class_names[row['class_id']][:30]})"
              f" -> pred={r['class_id']:2d} ({r['class_name'][:30]}) "
              f"conf={r['confidence']:.3f}")


if __name__ == "__main__":
    main()
