"""Tests for detection<->ground-truth matching and metric computation."""

from __future__ import annotations

import pytest

from src.detector import Detection
from src.evaluation import (
    DetectionMetrics,
    ImageResult,
    evaluate_image,
    match_detections,
)


def test_matching_counts_tp_fp_fn():
    preds = [Detection(box=(0, 0, 100, 200)), Detection(box=(500, 500, 100, 200))]
    gt = [(0, 0, 100, 200), (300, 0, 100, 200)]
    matches, unmatched_preds, unmatched_gts = match_detections(preds, gt, 0.5)
    assert len(matches) == 1            # first prediction matches first GT
    assert unmatched_preds == [1]       # second prediction is a false positive
    assert unmatched_gts == [1]         # second GT is missed (false negative)

    result = evaluate_image(preds, gt, "synthetic.png")
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 1)


def test_matching_is_one_to_one():
    """Two predictions overlapping one GT yield exactly one TP."""
    preds = [Detection(box=(0, 0, 100, 200)), Detection(box=(2, 2, 100, 200))]
    gt = [(0, 0, 100, 200)]
    matches, unmatched_preds, unmatched_gts = match_detections(preds, gt, 0.5)
    assert len(matches) == 1
    assert len(unmatched_preds) == 1
    assert unmatched_gts == []


def test_low_iou_prediction_is_false_positive():
    """A prediction with IoU < 0.5 must not count as a TP."""
    # Offset by 60 of 100 px width -> IoU = 0.25
    preds = [Detection(box=(60, 0, 100, 200))]
    gt = [(0, 0, 100, 200)]
    result = evaluate_image(preds, gt, "synthetic.png")
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_empty_predictions_all_gt_are_fn():
    result = evaluate_image([], [(0, 0, 50, 100), (100, 0, 50, 100)], "empty.png")
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 2)
    assert result.pred_count == 0
    assert result.gt_count == 2


def test_empty_ground_truth_all_predictions_are_fp():
    result = evaluate_image([Detection(box=(0, 0, 50, 100))], [], "neg.png")
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 0)


def test_both_empty_is_zero():
    result = evaluate_image([], [], "nothing.png")
    assert result.gt_count == 0 and result.pred_count == 0
    assert result.true_positives == 0


def test_mean_matched_iou_of_perfect_match():
    result = evaluate_image([Detection(box=(0, 0, 50, 100))], [(0, 0, 50, 100)], "perfect.png")
    assert result.mean_matched_iou == pytest.approx(1.0)


def test_greedy_takes_highest_iou_pair_first():
    """Best pair must win even when listed second."""
    preds = [Detection(box=(0, 0, 100, 200)), Detection(box=(0, 0, 104, 200))]
    gt = [(0, 0, 100, 200)]
    matches, _, _ = match_detections(preds, gt, 0.5)
    best_pred = matches[0][0]
    best_iou = matches[0][2]
    assert best_pred == 0          # exact box wins over the slightly shifted one
    assert best_iou == pytest.approx(1.0)


def test_detection_metrics_aggregation():
    m = DetectionMetrics()
    m.add_image_result(ImageResult("a.png", gt_count=2, pred_count=2,
                                   true_positives=1, false_positives=1,
                                   false_negatives=1, matched_ious=[0.8]))
    m.add_image_result(ImageResult("b.png", gt_count=0, pred_count=1,
                                   true_positives=0, false_positives=1,
                                   false_negatives=0))
    assert m.images_evaluated == 2
    assert m.total_gt == 2 and m.total_predictions == 3
    assert m.true_positives == 1 and m.false_positives == 2 and m.false_negatives == 1
    assert m.precision == pytest.approx(1 / 3)
    assert m.recall == pytest.approx(1 / 2)
    assert m.f1 == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))
    assert m.mean_matched_iou == pytest.approx(0.8)


def test_precision_recall_f1_zero_denominators_are_safe():
    m = DetectionMetrics()
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0
    m2 = DetectionMetrics()
    m2.add_image_result(ImageResult("neg.png", 0, 1, 0, 1, 0))  # only FPs
    assert m2.recall == 0.0
    assert m2.f1 == 0.0


def test_csv_row_shape():
    row = ImageResult("img.png", 3, 2, 2, 0, 1, [0.9, 0.7]).as_csv_row()
    assert row["image_name"] == "img.png"
    assert row["ground_truth_count"] == 3
    assert row["prediction_count"] == 2
    assert row["true_positives"] == 2
    assert row["false_positives"] == 0
    assert row["false_negatives"] == 1
    assert row["mean_matched_iou"] == pytest.approx(0.8)
