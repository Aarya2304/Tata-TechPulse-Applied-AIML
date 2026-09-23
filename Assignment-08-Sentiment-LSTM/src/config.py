"""Central configuration for the LSTM sentiment-classification project.

All paths are relative to this assignment's project root (no absolute paths).
"""

from __future__ import annotations

import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
RAW_DATA_PATH = DOWNLOAD_DIR / "reviews_Automotive_5.json.gz"  # gzipped JSON-lines

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

MODEL_PATH = MODELS_DIR / "best_sentiment_lstm.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_CSV = ARTIFACTS_DIR / "classification_report.csv"
SAMPLE_PREDICTIONS_CSV = ARTIFACTS_DIR / "sample_predictions.csv"
TRAINING_LOG_CSV = REPORTS_DIR / "training_log.csv"

# ---------------------------------------------------------------------------
# Dataset (Amazon Automotive product reviews, SNAP/UCSD McAuley lab)
# ---------------------------------------------------------------------------
DATASET_NAME = "Amazon Automotive product reviews (McAuley SNAP, 2016)"
DATASET_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/"
    "categoryFiles/reviews_Automotive_5.json.gz"
)
TEXT_COLUMN = "reviewText"     # raw review text
TITLE_COLUMN = "summary"       # review headline (prepended to the text)
RATING_COLUMN = "overall"      # 1..5 stars
LABEL_COLUMN = "sentiment"     # derived: 1 = positive (4-5 stars), 0 = negative (1-2 stars)
CLASS_NAMES = {0: "Negative", 1: "Positive"}
NEUTRAL_RATING = 3  # 3-star reviews are treated as neutral and dropped

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.10
VAL_FRACTION = 0.10  # of the full dataset; remainder is training

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
VOCAB_SIZE = 12000       # maximum vocabulary (incl. 0=pad and 1=<OOV>)
SEQUENCE_LENGTH = 120    # fixed padded length per review (covers the median 52 words)

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT = 0.4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 12
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3
# Calmer decay than patience=1: with patience=1 the LR halved almost every
# epoch and learning stalled early (measured); patience=2 lets val_loss
# plateau briefly before decaying.
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MONITOR_METRIC = "val_loss"

# ---------------------------------------------------------------------------
# Artifacts / reporting
# ---------------------------------------------------------------------------
MAX_SAMPLE_PREDICTIONS = 12   # rows kept in sample_predictions.csv (half wrong, half right)
PLOT_DPI = 120
