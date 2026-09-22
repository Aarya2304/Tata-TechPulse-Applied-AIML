"""Central configuration for the traffic-sign CNN project."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
TRAIN_IMAGES_DIR = DOWNLOADS_DIR / "GTSRB" / "Final_Training" / "Images"
TEST_GT_CSV = DOWNLOADS_DIR / "GTSRB" / "Final_Test" / "GT-final_test.csv"
TEST_IMAGES_DIR = DOWNLOADS_DIR / "GTSRB" / "Final_Test" / "Images"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_traffic_sign_cnn.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
REPORT_CSV_PATH = ARTIFACTS_DIR / "classification_report.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Dataset (official GTSRB archive hosted for the benchmark organisers)
# ---------------------------------------------------------------------------
GTSRB_URLS = {
    "train": ("https://sid.erda.dk/public/archives/"
              "daaeac0d7ce1152aea9b61d9f1e19370/"
              "GTSRB_Final_Training_Images.zip"),
    "test_images": ("https://sid.erda.dk/public/archives/"
                    "daaeac0d7ce1152aea9b61d9f1e19370/"
                    "GTSRB_Final_Test_Images.zip"),
    "test_gt": ("https://sid.erda.dk/public/archives/"
                "daaeac0d7ce1152aea9b61d9f1e19370/"
                "GTSRB_Final_Test_GT.zip"),
}

NUM_CLASSES = 43

# Official GTSRB class names (class id -> human-readable name).
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles",
    "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

# ---------------------------------------------------------------------------
# Images / training
# ---------------------------------------------------------------------------
IMG_SIZE = (32, 32)          # (height, width); GTSRB is variable-size
CHANNELS = 3
VALIDATION_FRACTION = 0.15   # stratified carve-out from official train set
BATCH_SIZE = 64
EPOCHS = 30
AUGMENT_TRAINING = True

# Light augmentation only: traffic signs have meaningful orientations, so
# no flips. Small rotation/translation/zoom are applied in preprocessing.
AUGMENTATION = {
    "rotation": 0.06,        # fraction of 2*pi  (~ +/- 11 degrees)
    "translation": 0.06,     # fraction of image side
    "zoom": 0.08,
}
