"""Tests for dataset discovery/validation/split (fast, no real download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (class_distribution, discover_training_data,
                             load_test_dataframe, stratified_split,
                             validate_classes)


# ---------------------------------------------------------------------------
# Synthetic GTSRB-style tree (tiny) so tests never download the real data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("downloads")
    train_root = root / "GTSRB" / "Final_Training" / "Images"
    test_root = root / "GTSRB" / "Final_Test"
    n_classes = 4
    for cls in range(n_classes):
        cdir = train_root / f"{cls:05d}"
        cdir.mkdir(parents=True)
        for i in range(10 + cls):          # 10, 11, 12, 13 per class
            _write_ppm(cdir / f"{cls:05d}_{i:05d}.ppm",
                       size=(20 + i, 18 + i))
    test_root.mkdir(parents=True)
    (test_root / "Images").mkdir()
    rows = []
    for i in range(12):
        _write_ppm(test_root / "Images" / f"{i:05d}.ppm", size=(24, 24))
        rows.append({"Filename": f"{i:05d}.ppm", "Width": 24, "Height": 24,
                     "Roi.X1": 1, "Roi.Y1": 1, "Roi.X2": 20, "Roi.Y2": 20,
                     "ClassId": i % n_classes})
    pd.DataFrame(rows).to_csv(test_root / "GT-final_test.csv", sep=";",
                              index=False)
    return root


def _write_ppm(path, size):
    """Write a valid binary PPM (P6) image of the given (w, h)."""
    w, h = size
    header = f"P6\n{w} {h}\n255\n".encode()
    body = bytes(np.arange(h * w * 3, dtype=np.uint8) % 255)
    path.write_bytes(header + body)


@pytest.fixture()
def use_synthetic(synthetic_dataset, monkeypatch):
    from src import config
    gtsrb_root = synthetic_dataset / "GTSRB"
    monkeypatch.setattr(config, "DOWNLOADS_DIR", synthetic_dataset)
    monkeypatch.setattr(config, "TRAIN_IMAGES_DIR",
                        gtsrb_root / "Final_Training" / "Images")
    monkeypatch.setattr(config, "TEST_IMAGES_DIR",
                        gtsrb_root / "Final_Test" / "Images")
    monkeypatch.setattr(config, "TEST_GT_CSV",
                        gtsrb_root / "Final_Test" / "GT-final_test.csv")
    return synthetic_dataset


# ---------------------------------------------------------------------------
def test_validate_classes_accepts_valid():
    validate_classes([0, 1, 42])            # must not raise


def test_validate_classes_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_classes([0, 43])
    with pytest.raises(ValueError):
        validate_classes([-1])


def test_class_distribution_counts():
    dist = class_distribution([0, 0, 2, 1, 2, 2])
    assert dist == {0: 2, 1: 1, 2: 3}


def test_discover_training_data(use_synthetic):
    paths, labels, counts = discover_training_data()
    assert len(paths) == 10 + 11 + 12 + 13
    assert sorted(counts) == [0, 1, 2, 3]
    assert counts[0] == 10 and counts[3] == 13
    assert set(labels) == {0, 1, 2, 3}
    # deterministic ordering: sorted paths, labels aligned
    assert paths == sorted(paths)
    assert labels[0] == 0


def test_discover_missing_dir_raises(use_synthetic, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "TRAIN_IMAGES_DIR",
                        use_synthetic / "does" / "not" / "exist")
    with pytest.raises(FileNotFoundError):
        discover_training_data()


def test_load_test_dataframe(use_synthetic):
    df = load_test_dataframe()
    assert len(df) == 12
    assert list(df.columns) == ["image_path", "class_id"]
    assert set(df["class_id"]) == {0, 1, 2, 3}
    assert df["image_path"].iloc[0].endswith("00000.ppm")


def test_load_test_dataframe_missing_columns(tmp_path, use_synthetic,
                                             monkeypatch):
    from src import config
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Filename": ["x.ppm"]}).to_csv(bad, sep=";", index=False)
    monkeypatch.setattr(config, "TEST_GT_CSV", bad)
    with pytest.raises(ValueError):
        load_test_dataframe()


def test_stratified_split_preserves_classes(use_synthetic):
    paths, labels, _ = discover_training_data()
    tr_p, tr_l, va_p, va_l = stratified_split(paths, labels,
                                              validation_fraction=0.25)
    assert len(tr_p) + len(va_p) == len(paths)
    assert len(tr_l) == len(tr_p) and len(va_l) == len(va_p)
    # every class present in both sides
    assert set(tr_l) == {0, 1, 2, 3}
    assert set(va_l) == {0, 1, 2, 3}
    # deterministic for fixed seed
    tr_p2, _, va_p2, _ = stratified_split(paths, labels,
                                          validation_fraction=0.25)
    assert tr_p == tr_p2 and va_p == va_p2


def test_real_class_names_consistent():
    """The shipped class-name mapping must have exactly 43 entries."""
    assert len(config.CLASS_NAMES) == config.NUM_CLASSES == 43
