"""GTSRB dataset discovery, download, validation and split logic.

Layout produced by the official zips (see data/README.md):

    data/downloads/GTSRB/Final_Training/Images/000xx/*.ppm + GT-000xx.csv
    data/downloads/GTSRB/Final_Test/Images/*.ppm
    data/downloads/GTSRB/Final_Test/GT-final_test.csv   (Filename;...;ClassId)

All functions read paths from ``src.config`` at call time so tests can
monkeypatch them to a tiny synthetic dataset.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config


# ---------------------------------------------------------------------------
# Download / extraction
# ---------------------------------------------------------------------------
def ensure_downloaded() -> None:
    """Download and extract the official GTSRB archives if missing."""
    if config.TRAIN_IMAGES_DIR.exists() and config.TEST_GT_CSV.exists():
        return
    import shutil
    import urllib.request

    config.DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in config.GTSRB_URLS.items():
        dest = config.DOWNLOADS_DIR / f"{name}.zip"
        print(f"[data_loader] downloading {name} archive...")
        with urllib.request.urlopen(url, timeout=300) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(config.DOWNLOADS_DIR)
        dest.unlink(missing_ok=True)
    # The test-labels zip extracts GT-final_test.csv next to GTSRB/;
    # move it where the pipeline expects it.
    gt_src = config.DOWNLOADS_DIR / "GT-final_test.csv"
    if gt_src.exists():
        shutil.move(str(gt_src), str(config.TEST_GT_CSV))
    if not (config.TRAIN_IMAGES_DIR.exists() and config.TEST_GT_CSV.exists()):
        raise RuntimeError("GTSRB download/extraction failed")


# ---------------------------------------------------------------------------
# Discovery / validation
# ---------------------------------------------------------------------------
def discover_training_data() -> tuple[list[str], list[int], dict[int, int]]:
    """Scan class folders; returns (paths, labels, per-class counts).

    Every image listed in a class's GT-*.csv must exist as a .ppm; the
    count of readable .ppm files is what defines the dataset. Sorted for
    deterministic ordering.
    """
    if not config.TRAIN_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Training images not found at {config.TRAIN_IMAGES_DIR}. "
            "Run ensure_downloaded() or place the GTSRB archive there.")
    paths: list[str] = []
    labels: list[int] = []
    counts: dict[int, int] = {}
    for class_dir in sorted(config.TRAIN_IMAGES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        images = sorted(class_dir.glob("*.ppm"))
        counts[class_id] = len(images)
        paths.extend(str(p) for p in images)
        labels.extend([class_id] * len(images))
    return paths, labels, counts


def load_test_dataframe() -> pd.DataFrame:
    """Official test labels -> DataFrame(image_path, class_id)."""
    if not config.TEST_GT_CSV.exists():
        raise FileNotFoundError(
            f"Test labels not found at {config.TEST_GT_CSV}")
    gt = pd.read_csv(config.TEST_GT_CSV, sep=";")
    required = {"Filename", "ClassId"}
    if not required.issubset(gt.columns):
        raise ValueError(f"Test GT missing columns: {required - set(gt.columns)}")
    gt["image_path"] = gt["Filename"].map(
        lambda f: str(config.TEST_IMAGES_DIR / f))
    gt = gt.rename(columns={"ClassId": "class_id"})
    return gt[["image_path", "class_id"]]


def validate_classes(labels: list[int], n_classes: int = config.NUM_CLASSES) -> None:
    """All labels must be integers within [0, n_classes)."""
    bad = [l for l in labels if not (0 <= l < n_classes)]
    if bad:
        raise ValueError(f"Labels outside [0, {n_classes}): {bad[:5]}...")


def class_distribution(labels: list[int]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for l in labels:
        dist[l] = dist.get(l, 0) + 1
    return dict(sorted(dist.items()))


# ---------------------------------------------------------------------------
# Split (stratified)
# ---------------------------------------------------------------------------
def stratified_split(paths: list[str], labels: list[int],
                     validation_fraction: float = config.VALIDATION_FRACTION
                     ) -> tuple[list[str], list[int], list[str], list[int]]:
    """Stratified train/validation split of the official training set.

    The official test set is used untouched as the test split (documented
    in the README), so only the training pool is carved here.
    """
    from sklearn.model_selection import train_test_split
    train_p, val_p, train_l, val_l = train_test_split(
        paths, labels, test_size=validation_fraction,
        random_state=config.RANDOM_STATE, stratify=labels)
    return train_p, train_l, val_p, val_l


# ---------------------------------------------------------------------------
# Single-image loading (tf)
# ---------------------------------------------------------------------------
def _decode_bytes_pil(raw) -> np.ndarray:
    """Decode raw image bytes with PIL (Pillow).

    GTSRB ships PPM files, which tf.io.decode_image does NOT support
    (JPEG/PNG/GIF/BMP/WebP only), so PIL is the decoder of record here.
    ``raw`` arrives from tf.numpy_function as a 0-d numpy array of bytes;
    normalise it to plain ``bytes`` before decoding.
    """
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.tobytes()
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def decode_image(path: tf.Tensor) -> tf.Tensor:
    """Read + decode (PIL) + resize to IMG_SIZE; returns uint8 [H,W,3]."""
    raw = tf.io.read_file(path)
    img = tf.numpy_function(_decode_bytes_pil, [raw], tf.uint8)
    img.set_shape([None, None, config.CHANNELS])
    img = tf.image.resize_with_pad(tf.cast(img, tf.float32),
                                   *config.IMG_SIZE)
    return tf.cast(tf.round(img), tf.uint8)


def load_and_decode(path: str) -> tf.Tensor:
    """Convenience wrapper for single paths (tests / prediction)."""
    return decode_image(tf.constant(str(path)))
