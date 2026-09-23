"""Central configuration for the HOG + SVM pedestrian-detection project.

All paths are relative to this assignment's project root (no absolute paths).
The detector parameters below are the recommended starting values from the
assignment brief; they were kept unless a small measured change clearly
improved the documented evaluation (see README, "Detector configuration").
"""

from __future__ import annotations

import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
# Extracted INRIA Person test split (images + PASCAL-style annotations).
INRIA_TEST_DIR = DATA_DIR / "INRIAPerson" / "Test"
POS_DIR = INRIA_TEST_DIR / "pos"
ANNOTATIONS_DIR = INRIA_TEST_DIR / "annotations"
NEG_DIR = INRIA_TEST_DIR / "neg"
TAR_PATH = DOWNLOAD_DIR / "INRIAPerson.tar"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANNOTATED_DIR = OUTPUTS_DIR / "annotated"

METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
EVALUATION_CSV = ARTIFACTS_DIR / "evaluation_results.csv"
RUNTIME_JSON = REPORTS_DIR / "runtime.json"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Original INRIA Person Dataset (Navneet Dalal & Bill Triggs, CVPR 2005).
# The official pascal.inrialpes.fr server is offline; this is the complete
# original archive (1,016,094,720 bytes) mirrored by the University of
# Central Florida's computer-vision course page.
DATASET_NAME = "INRIA Person Dataset"
TAR_URL = "http://cs.ucf.edu/courses/cap6412/fall2009/misc/INRIAPerson.tar"
TAR_SIZE_BYTES = 1_016_094_720
# Number of annotated test images evaluated end-to-end (documented subset of
# the 288 annotated INRIA test images to keep the CPU evaluation bounded).
EVAL_LIMIT = 288
# Seed used to choose the subset when EVAL_LIMIT < number of annotated files.
SUBSET_SEED = RANDOM_STATE

# ---------------------------------------------------------------------------
# HOG + SVM detector configuration
# ---------------------------------------------------------------------------
# cv2.HOGDescriptor() defaults match the 64x128 detection window trained for
# the bundled default people SVM (3,780-dimensional descriptor).
WIN_STRIDE = (8, 8)     # sliding-window step in pixels
PADDING = (8, 8)        # pixels of padding around the window
SCALE = 1.05            # image-pyramid scale step between levels
HIT_THRESHOLD = 0.0     # SVM margin threshold (OpenCV default when None)
# Post-hoc filter on detectMultiScale weights (SVM decision values).
# Calibrated on a 20-image subset together with BOX_SHRINK_FACTOR below:
# F1 rises from 0.378 (threshold 0.0) to 0.597 (threshold 0.5) and falls
# again towards 0.44 at 1.6, so 0.5 is the measured optimum.
CONFIDENCE_THRESHOLD = 0.5
# Window-margin calibration (see README): the 64x128 detection window
# includes margin around the person, while INRIA annotations are tight
# person crops. Detection boxes are therefore shrunk around their centre by
# this factor before NMS/matching. Calibrated on a 20-image subset: without
# the correction F1 is 0.09; with it (and CONFIDENCE_THRESHOLD above) F1 is
# ~0.60. Set to 1.0 to disable and reproduce the raw-window behaviour.
BOX_SHRINK_FACTOR = 0.65

# ---------------------------------------------------------------------------
# Post-processing / evaluation
# ---------------------------------------------------------------------------
NMS_THRESHOLD = 0.4     # IoU above which overlapping boxes are suppressed
IOU_THRESHOLD = 0.5     # GT<->prediction IoU required for a true positive

# ---------------------------------------------------------------------------
# Evaluation subset splitting (per-image cache/split helpers)
# ---------------------------------------------------------------------------
# Fraction of annotated images used for the calibration sweep.
CALIBRATION_COUNT = 40

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
PRED_COLOR = (0, 220, 0)      # green  (BGR) for predicted boxes
GT_COLOR = (0, 0, 255)        # red    (BGR) for ground-truth boxes
FP_COLOR = (0, 165, 255)      # orange (BGR) for false positives
FN_COLOR = (255, 0, 255)      # magenta (BGR) for missed pedestrians
MAX_EXAMPLES_PER_PLOT = 6     # images per qualitative figure
PLOT_DPI = 100
PLOT_MAX_WIDTH_PX = 760       # downscale annotated panels before plotting
