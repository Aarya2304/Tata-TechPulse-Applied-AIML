"""Tests for Non-Maximum Suppression."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import PedestrianDetector


def test_nms_removes_highly_overlapping_boxes():
    boxes = [(10, 10, 50, 100), (12, 11, 50, 100), (100, 100, 50, 100)]
    scores = [2.0, 3.0, 1.0]
    keep = PedestrianDetector.apply_nms(boxes, scores, 0.4)
    # Box 1 (higher score) survives; its near-duplicate box 0 is suppressed;
    # the disjoint box 2 is kept.
    assert sorted(keep) == [1, 2]


def test_nms_keeps_separate_boxes():
    boxes = [(0, 0, 50, 100), (200, 0, 50, 100), (0, 300, 50, 100)]
    scores = [1.0, 0.9, 0.8]
    keep = PedestrianDetector.apply_nms(boxes, scores, 0.4)
    assert sorted(keep) == [0, 1, 2]


def test_nms_empty_input():
    assert PedestrianDetector.apply_nms([], [], 0.4) == []


def test_nms_suppresses_identical_duplicate():
    boxes = [(10, 10, 64, 128), (10, 10, 64, 128)]
    scores = [0.5, 1.5]
    keep = PedestrianDetector.apply_nms(boxes, scores, 0.4)
    assert keep == [1]


def test_nms_borderline_iou_respected():
    """Suppression must follow the threshold on known IoU values."""
    # Two 100x50 boxes offset by 60 px horizontally:
    # intersection = 40*50 = 2000, union = 2*5000 - 2000 = 8000 -> IoU 0.25
    boxes = [(0, 0, 100, 50), (60, 0, 100, 50)]
    # threshold 0.2: 0.25 > 0.2 -> the lower-scoring box is suppressed
    keep = PedestrianDetector.apply_nms(boxes, [2.0, 1.0], 0.2)
    assert keep == [0]
    # threshold 0.3: 0.25 < 0.3 -> both boxes survive
    keep2 = PedestrianDetector.apply_nms(boxes, [2.0, 1.0], 0.3)
    assert sorted(keep2) == [0, 1]


def test_nms_scores_length_mismatch_raises():
    with pytest.raises(ValueError):
        PedestrianDetector.apply_nms([(0, 0, 10, 10)], [1.0, 2.0], 0.4)


def test_nms_orders_kept_boxes_by_descending_score():
    boxes = [(0, 0, 10, 10), (100, 100, 10, 10), (200, 200, 10, 10)]
    scores = [0.3, 2.2, 1.4]
    keep = PedestrianDetector.apply_nms(boxes, scores, 0.4)
    assert keep == [1, 2, 0]
    assert isinstance(keep[0], int)


def test_nms_matches_bruteforce_reference():
    """Cross-check the vectorised NMS against a naive implementation."""
    boxes = [(int(x), int(y), 40, 90) for x, y in
             np.random.default_rng(7).integers(0, 150, (30, 2))]
    scores = list(np.random.default_rng(7).uniform(0, 3, 30))
    keep = set(PedestrianDetector.apply_nms(boxes, scores, 0.4))

    # naive reference
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    expected = []
    removed = [False] * len(boxes)

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    for i in order:
        if removed[i]:
            continue
        expected.append(i)
        for j in order:
            if j == i or removed[j]:
                continue
            if iou(boxes[i], boxes[j]) > 0.4:
                removed[j] = True
    assert keep == set(expected)
