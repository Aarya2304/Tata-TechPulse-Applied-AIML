"""Evaluation: IoU-based matching, TP/FP/FN accumulation and detection metrics.

Matching protocol (greedy, documented in the README)
----------------------------------------------------
1. Compute IoU between every prediction and every ground-truth box.
2. Repeatedly take the unmatched prediction/GT pair with the highest IoU.
3. A pair is a true positive when its IoU >= ``config.IOU_THRESHOLD`` (0.5);
   each ground-truth box and each prediction is matched at most once.
4. Remaining unmatched predictions are false positives; remaining unmatched
   ground-truth boxes are false negatives.

Precision = TP / (TP + FP), Recall = TP / (TP + FN),
F1 = 2 * P * R / (P + R). All zero denominators yield 0.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src import config
from src.detector import Box, Detection, iou


def match_detections(
    predictions: list[Detection],
    gt_boxes: list[Box],
    iou_threshold: float = config.IOU_THRESHOLD,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy IoU matching between predictions and ground truth.

    Returns ``(matches, unmatched_preds, unmatched_gt)`` where each match is
    ``(pred_index, gt_index, iou_value)``.
    """
    pairs: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(predictions):
        for g_idx, gt in enumerate(gt_boxes):
            overlap = iou(pred.box, gt)
            if overlap >= iou_threshold:
                pairs.append((overlap, p_idx, g_idx))
    # Highest IoU first; stable ties keep deterministic ordering.
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, p_idx, g_idx in pairs:
        if p_idx in matched_preds or g_idx in matched_gts:
            continue
        matched_preds.add(p_idx)
        matched_gts.add(g_idx)
        matches.append((p_idx, g_idx, overlap))
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


@dataclass
class ImageResult:
    """Per-image evaluation record."""

    image_name: str
    gt_count: int
    pred_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: list[float] = field(default_factory=list)

    @property
    def mean_matched_iou(self) -> float:
        return sum(self.matched_ious) / len(self.matched_ious) if self.matched_ious else 0.0

    def as_csv_row(self) -> dict[str, float | int | str]:
        return {
            "image_name": self.image_name,
            "ground_truth_count": self.gt_count,
            "prediction_count": self.pred_count,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "mean_matched_iou": round(self.mean_matched_iou, 4),
        }


@dataclass
class DetectionMetrics:
    """Aggregated metrics over a set of evaluated images."""

    images_evaluated: int = 0
    total_gt: int = 0
    total_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_ious: list[float] = field(default_factory=list)
    per_image: list[ImageResult] = field(default_factory=list)

    def add_image_result(self, result: ImageResult) -> None:
        self.images_evaluated += 1
        self.total_gt += result.gt_count
        self.total_predictions += result.pred_count
        self.true_positives += result.true_positives
        self.false_positives += result.false_positives
        self.false_negatives += result.false_negatives
        self.matched_ious.extend(result.matched_ious)
        self.per_image.append(result)

    # -- aggregate metric helpers -------------------------------------
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_matched_iou(self) -> float:
        return (
            sum(self.matched_ious) / len(self.matched_ious) if self.matched_ious else 0.0
        )

    def summary(self) -> dict[str, float | int]:
        return {
            "images_evaluated": self.images_evaluated,
            "total_ground_truth_pedestrians": self.total_gt,
            "total_predictions": self.total_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1, 4),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "iou_threshold": config.IOU_THRESHOLD,
        }


def evaluate_image(
    detections: list[Detection],
    gt_boxes: list[Box],
    image_name: str = "",
    iou_threshold: float = config.IOU_THRESHOLD,
) -> ImageResult:
    """Evaluate one image: match detections to GT and count TP/FP/FN."""
    matches, unmatched_preds, unmatched_gts = match_detections(
        detections, gt_boxes, iou_threshold
    )
    return ImageResult(
        image_name=image_name,
        gt_count=len(gt_boxes),
        pred_count=len(detections),
        true_positives=len(matches),
        false_positives=len(unmatched_preds),
        false_negatives=len(unmatched_gts),
        matched_ious=[overlap for _, _, overlap in matches],
    )


def save_metrics(metrics: DetectionMetrics, path=None) -> dict:
    """Persist aggregate metrics as JSON and return the summary dict."""
    path = path or config.METRICS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = metrics.summary()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def save_per_image_csv(metrics: DetectionMetrics, path=None) -> None:
    """Persist the per-image evaluation table as CSV."""
    import csv

    path = path or config.EVALUATION_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_csv_row() for r in metrics.per_image]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
