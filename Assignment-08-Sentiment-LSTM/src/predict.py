"""Inference: load the trained LSTM and predict sentiment for raw sentences.

The preprocessing here is **exactly** the training preprocessing: the saved
fitted tokenizer (models/tokenizer.json) and the saved sequence length from
models/model_metadata.json are used — no re-fitting, no re-cleaning choices.

Usage:
    python -m src.predict "The vehicle is excellent and very comfortable."
    python -m src.predict --demo          # run built-in example sentences
    python -m src.predict --batch texts.txt   # one sentence per line
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src import config
from src.preprocessing import clean_text, load_tokenizer
from src.train_utils import pad_texts

DEFAULT_DEMO_TEXTS = [
    "The vehicle is excellent and very comfortable.",
    "This product is not good at all, it broke after two days.",
    "Works great on my car, easy to install and great value.",
    "Terrible quality, I would never buy this again.",
    "It is okay, does the job but nothing special.",
    "Absolutely love it, best purchase for my truck this year!",
]


def load_artifacts(model_path=None, tokenizer_path=None, metadata_path=None):
    """Load model, tokenizer and metadata; return (model, tokenizer, metadata)."""
    from tensorflow.keras.models import load_model

    model_path = model_path or config.MODEL_PATH
    tokenizer = load_tokenizer(tokenizer_path or config.TOKENIZER_PATH)
    with open(metadata_path or config.METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    model = load_model(model_path)
    return model, tokenizer, metadata


def predict_sentiment(text: str, model, tokenizer, metadata) -> dict:
    """Predict sentiment for one raw sentence using saved artifacts."""
    prob_array = model.predict(pad_texts([text], tokenizer, metadata), verbose=0)
    prob = float(np.asarray(prob_array).reshape(-1)[0])
    label = int(prob >= 0.5)
    return {
        "text": text,
        "probability": prob,
        "label": label,
        "sentiment": config.CLASS_NAMES[label],
        "confidence": prob if label == 1 else 1.0 - prob,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM vehicle-feedback sentiment classifier")
    parser.add_argument("text", nargs="*", help="sentence(s) to classify")
    parser.add_argument("--demo", action="store_true", help="run built-in demo sentences")
    parser.add_argument("--batch", type=str, default=None,
                        help="path to a text file with one sentence per line")
    args = parser.parse_args()

    model, tokenizer, metadata = load_artifacts()

    if args.demo:
        texts = DEFAULT_DEMO_TEXTS
    elif args.batch:
        with open(args.batch, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        parser.error("provide a sentence, --demo or --batch")

    for text in texts:
        result = predict_sentiment(text, model, tokenizer, metadata)
        print(
            f"{result['sentiment']:>8}  p={result['probability']:.3f}  |  {text}"
        )


if __name__ == "__main__":
    main()
