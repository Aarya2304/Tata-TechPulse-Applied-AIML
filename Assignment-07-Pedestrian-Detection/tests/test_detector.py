"""Tests for the HOG + SVM detector wrapper (synthetic images only)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.detector import Detection, PedestrianDetector, annotate_image


@pytest.fixture(scope="module")
def detector() -> PedestrianDetector:
    return PedestrianDetector()


def test_hog_detector_initializes_with_default_people_svm(detector):
    """The detector must use cv2's pretrained default people-detection SVM."""
    assert detector.hog is not None
    assert detector.hog.getDescriptorSize() == 3780
    assert detector.hog.winSize == (64, 128)


def test_detect_accepts_valid_synthetic_image(detector):
    """detect() runs on a plain synthetic image and returns a list."""
    img = np.full((240, 320, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (60, 20), (110, 200), (90, 60, 40), -1)  # torso-ish blob
    detections = detector.detect(img)
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detection)
        assert len(det.box) == 4


def test_detect_handles_empty_and_tiny_images(detector):
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    assert detector.detect(empty) == []
    tiny = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert detector.detect(tiny) == []


def test_confidence_threshold_filters_detections(detector):
    """Raising the confidence threshold must not increase detection count."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    rects, weights = detector.detect_raw(img)
    loose = detector.detect(img)
    detector_strict = PedestrianDetector(confidence_threshold=10.0)
    strict = detector_strict.detect(img)
    assert len(strict) <= len(loose)
    for det in strict:
        assert det.score > 10.0
    # raw outputs stay aligned
    assert len(rects) == len(weights)


def test_detect_raw_returns_numpy_weights(detector):
    img = np.full((128, 64, 3), 100, dtype=np.uint8)
    rects, weights = detector.detect_raw(img)
    assert isinstance(rects, list)
    assert weights.ndim == 1
    assert len(rects) == len(weights)


def test_annotate_image_draws_boxes_and_returns_copy():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = [Detection(box=(10, 10, 50, 100), score=1.23)]
    out = annotate_image(img, dets, gt_boxes=[(90, 10, 50, 100)])
    assert out.shape == img.shape
    assert out is not img  # must not mutate the input
    # some pixels must have been drawn (green channel on prediction border)
    assert out[:, :, 1].sum() > 0


def test_apply_nms_returns_sorted_valid_indices():
    boxes = [(0, 0, 50, 100), (5, 5, 50, 100), (200, 200, 50, 100)]
    scores = [1.0, 2.5, 0.5]
    keep = PedestrianDetector.apply_nms(boxes, scores, 0.4)
    assert keep[0] == 1  # highest score first
    assert all(0 <= i < len(boxes) for i in keep)
    assert len(set(keep)) == len(keep)
