"""Tests for the IoU implementation."""

from __future__ import annotations

import pytest

from src.detector import iou


def test_iou_exact_overlap_is_one():
    assert iou((10, 10, 50, 100), (10, 10, 50, 100)) == pytest.approx(1.0)


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_touching_edges_is_zero():
    """Boxes sharing only an edge have zero intersection area."""
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (0, 10, 10, 10)) == 0.0


def test_iou_partial_overlap_is_correct():
    """Half-overlapping 100x100 boxes -> IoU = 1/3."""
    got = iou((0, 0, 100, 100), (50, 0, 100, 100))
    assert got == pytest.approx(1 / 3)


def test_iou_containment_is_correct():
    """A 10x10 box fully inside a 100x100 box -> IoU = 100/10000 = 0.01."""
    got = iou((0, 0, 100, 100), (10, 10, 10, 10))
    assert got == pytest.approx(0.01)


def test_iou_symmetry():
    a, b = (0, 0, 100, 100), (50, 0, 100, 100)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_iou_degenerate_zero_width_box_is_safe():
    assert iou((0, 0, 0, 100), (0, 0, 50, 100)) == 0.0


def test_iou_degenerate_negative_size_box_is_safe():
    assert iou((0, 0, -5, 100), (0, 0, 50, 100)) == 0.0


def test_iou_both_degenerate_is_safe():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
