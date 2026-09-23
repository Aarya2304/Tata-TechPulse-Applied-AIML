"""HOG + SVM pedestrian detector (OpenCV) with Non-Maximum Suppression.

The detector is the classical Dalal & Triggs pipeline:

    image -> HOG features on a sliding window -> pretrained linear SVM
          -> candidate bounding boxes + scores -> confidence filter
          -> Non-Maximum Suppression -> final detections

Specifically, it uses ``cv2.HOGDescriptor`` together with
``cv2.HOGDescriptor_getDefaultPeopleDetector()`` — the pretrained
people-detection SVM bundled with OpenCV (trained on the INRIA Person
training set). No neural network and no other detection framework is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src import config

Box = tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class Detection:
    """A single pedestrian detection (or ground-truth box) in (x, y, w, h)."""

    box: Box
    score: float = 0.0


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-Union between two (x, y, w, h) boxes.

    Degenerate boxes (zero/negative width or height) are handled safely:
    the intersection is 0, so the IoU is 0.0 unless *both* boxes are
    degenerate in exactly the same way, which also yields 0.0.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:  # pragma: no cover - guarded by degenerate check
        return 0.0
    return float(inter_area) / float(union_area)


class PedestrianDetector:
    """Sliding-window HOG pedestrian detector using OpenCV's default SVM."""

    def __init__(
        self,
        win_stride: tuple[int, int] = config.WIN_STRIDE,
        padding: tuple[int, int] = config.PADDING,
        scale: float = config.SCALE,
        hit_threshold: float = config.HIT_THRESHOLD,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.NMS_THRESHOLD,
        box_shrink_factor: float = config.BOX_SHRINK_FACTOR,
    ) -> None:
        self.win_stride = tuple(win_stride)
        self.padding = tuple(padding)
        self.scale = float(scale)
        self.hit_threshold = float(hit_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.box_shrink_factor = float(box_shrink_factor)

        self.hog = cv2.HOGDescriptor()
        # OpenCV's pretrained people-detection SVM (linear SVM over the
        # 3,780-dimensional HOG descriptor of a 64x128 detection window).
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_raw(self, image: np.ndarray) -> tuple[list[Box], np.ndarray]:
        """Run the raw sliding-window detector.

        Returns ``(rects, weights)`` exactly as produced by
        ``hog.detectMultiScale`` — heavily overlapping candidate boxes are
        expected at this stage and are pruned later by NMS.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return [], np.empty((0,), dtype=float)
        # The detector slides a fixed 64x128 window; images smaller than the
        # window cannot contain even one detection window.
        h, w = image.shape[:2]
        if h < 128 or w < 64:
            return [], np.empty((0,), dtype=float)
        rects, weights = self.hog.detectMultiScale(
            image,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=self.hit_threshold,
            groupThreshold=0,  # disable OpenCV grouping; NMS below does this job
        )
        rects = [tuple(int(v) for v in r) for r in np.asarray(rects).reshape(-1, 4)]
        weights = np.asarray(weights, dtype=float).reshape(-1)
        return rects, weights

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Full pipeline: raw detections -> confidence filter -> NMS."""
        rects, weights = self.detect_raw(image)
        keep_idx = [i for i, w in enumerate(weights) if w > self.confidence_threshold]
        boxes = [self._shrink_box(rects[i]) for i in keep_idx]
        scores = weights[keep_idx]
        keep = self.apply_nms(boxes, scores, self.nms_threshold)
        return [Detection(box=boxes[i], score=float(scores[i])) for i in keep]

    def _shrink_box(self, box: Box) -> Box:
        """Shrink a detection window around its centre by ``box_shrink_factor``.

        OpenCV's 64x128 detection window includes margin around the person,
        while pedestrian annotations (INRIA and most datasets) are tight
        person crops. Reporting the raw window therefore inflates boxes by a
        constant factor and depresses IoU against tight ground truth. This
        calibration is a standard remedy for that window-margin bias.
        """
        factor = self.box_shrink_factor
        if factor >= 1.0 or factor <= 0.0:
            return box
        x, y, w, h = box
        new_w, new_h = int(round(w * factor)), int(round(h * factor))
        return (int(x + (w - new_w) / 2), int(y + (h - new_h) / 2), new_w, new_h)

    # ------------------------------------------------------------------
    # Non-Maximum Suppression
    # ------------------------------------------------------------------
    @staticmethod
    def apply_nms(
        boxes: list[Box], scores: list[float] | np.ndarray, nms_threshold: float
    ) -> list[int]:
        """Greedy Non-Maximum Suppression on (x, y, w, h) boxes.

        1. Sort boxes by score (descending).
        2. Take the highest-scoring remaining box and suppress every other
           box whose IoU with it exceeds ``nms_threshold``.
        3. Repeat until no boxes remain.

        Returns the indices (into the input lists) of the kept boxes,
        ordered by descending score. Boxes with IoU exactly equal to the
        threshold are *not* suppressed (strict ``>`` comparison).
        """
        if len(boxes) == 0:
            return []
        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        if scores_arr.size != len(boxes):
            raise ValueError("scores and boxes must have the same length")

        order = np.argsort(-scores_arr, kind="stable")
        boxes_arr = np.asarray(boxes, dtype=float).reshape(-1, 4)

        areas = boxes_arr[:, 2] * boxes_arr[:, 3]
        keep: list[int] = []
        suppressed = np.zeros(len(boxes), dtype=bool)
        for pos in order:
            if suppressed[pos]:
                continue
            keep.append(int(pos))
            # Vectorised IoU of every remaining box against the kept box.
            x1 = np.maximum(boxes_arr[pos, 0], boxes_arr[:, 0])
            y1 = np.maximum(boxes_arr[pos, 1], boxes_arr[:, 1])
            x2 = np.minimum(boxes_arr[pos, 0] + boxes_arr[pos, 2],
                            boxes_arr[:, 0] + boxes_arr[:, 2])
            y2 = np.minimum(boxes_arr[pos, 1] + boxes_arr[pos, 3],
                            boxes_arr[:, 1] + boxes_arr[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = areas + areas[pos] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                ious = np.where(union > 0, inter / union, 0.0)
            # Suppress boxes that overlap the kept box more than the
            # threshold (and are not the kept box itself).
            suppressed |= (ious > nms_threshold)
            suppressed[pos] = True
            if suppressed.all():
                break
        return keep


def annotate_image(
    image: np.ndarray,
    detections: list[Detection],
    gt_boxes: list[Box] | None = None,
    draw_scores: bool = True,
) -> np.ndarray:
    """Return a copy of *image* with predicted (and optional GT) boxes drawn.

    Predictions are green, ground truth is red. Confidence values are drawn
    as text labels above each predicted box.
    """
    canvas = image.copy()
    for det in detections:
        x, y, w, h = det.box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), config.PRED_COLOR, 2)
        if draw_scores:
            label = f"{det.score:.2f}"
            ty = max(12, y - 6)
            cv2.putText(canvas, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, config.PRED_COLOR, 1, cv2.LINE_AA)
    for box in gt_boxes or []:
        x, y, w, h = box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), config.GT_COLOR, 2)
    return canvas
