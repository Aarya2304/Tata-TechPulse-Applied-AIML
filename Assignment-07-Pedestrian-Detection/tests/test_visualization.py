"""Tests for visualization helpers (all on tiny synthetic images)."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from src.detector import Detection
from src.visualization import (
    plot_detection_examples,
    plot_false_positives,
    plot_hog_visualization,
    plot_missed_pedestrians,
)


@pytest.fixture
def synthetic_scene(tmp_path):
    """Small synthetic scene + one detection/GT pair."""
    img = np.full((160, 120, 3), 90, dtype=np.uint8)
    img_path = tmp_path / "scene.png"
    cv2.imwrite(str(img_path), img)
    return img_path, [Detection(box=(10, 10, 40, 80), score=0.87)], [(60, 10, 40, 80)]


def test_detection_examples_figure(synthetic_scene, tmp_path):
    path, dets, gt = synthetic_scene
    out = tmp_path / "fig_examples.png"
    result = plot_detection_examples([(str(path), dets, gt)], out_path=out, cols=1)
    assert result.exists() and result.stat().st_size > 1000


def test_false_positives_figure(synthetic_scene, tmp_path):
    path, dets, gt = synthetic_scene  # detection does not match GT -> an FP
    out = tmp_path / "fig_fp.png"
    result = plot_false_positives([(str(path), dets, gt)], out_path=out, cols=1)
    assert result.exists() and result.stat().st_size > 1000


def test_missed_pedestrians_figure(synthetic_scene, tmp_path):
    path, dets, gt = synthetic_scene  # GT unmatched -> an FN
    out = tmp_path / "fig_fn.png"
    result = plot_missed_pedestrians([(str(path), dets, gt)], out_path=out, cols=1)
    assert result.exists() and result.stat().st_size > 1000


def test_hog_visualization_figure(synthetic_scene, tmp_path):
    path, _, _ = synthetic_scene
    out = tmp_path / "fig_hog.png"
    from src.detector import PedestrianDetector

    result = plot_hog_visualization(path, PedestrianDetector(), out_path=out)
    assert result.exists() and result.stat().st_size > 1000


def test_annotated_output_written(tmp_path):
    """annotate_image + imwrite produce a real annotated file (pipeline step 10)."""
    from src.detector import annotate_image

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas = annotate_image(img, [Detection(box=(5, 5, 30, 60), score=1.1)], [(50, 5, 30, 60)])
    out = tmp_path / "annotated.png"
    assert cv2.imwrite(str(out), canvas)
    loaded = cv2.imread(str(out))
    assert loaded is not None and loaded.shape == (100, 100, 3)
    assert not np.array_equal(loaded, img)
