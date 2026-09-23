"""Qualitative visualizations: detection examples, FP/FN panels, HOG illustration."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.detector import Box, Detection, annotate_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _downscale(img: np.ndarray, max_width: int = config.PLOT_MAX_WIDTH_PX) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _panel_grid(
    panels: list[tuple[str, np.ndarray]],
    cols: int,
    out_path: Path,
    suptitle: str | None = None,
) -> None:
    """Render a list of (title, rgb-image) panels on a grid and save the figure."""
    if not panels:
        return
    cols = max(1, min(cols, len(panels)))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), dpi=config.PLOT_DPI)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (title, rgb) in zip(axes, panels):
        ax.imshow(rgb)
        ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if suptitle else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Detection examples (predictions vs ground truth)
# ---------------------------------------------------------------------------


def plot_detection_examples(
    examples: list[tuple[str, list[Detection], list[Box]]],
    out_path: Path = config.PLOTS_DIR / "detection_examples.png",
    cols: int = 3,
) -> Path:
    """Show several images with predicted (green) and GT (red) boxes.

    ``examples`` is a list of ``(image_name, detections, gt_boxes)``.
    """
    panels: list[tuple[str, np.ndarray]] = []
    for name, detections, gt_boxes in examples:
        img = _downscale(_load_bgr(name))
        annotated = annotate_image(img, detections, gt_boxes, draw_scores=True)
        title = f"{Path(name).name}  pred={len(detections)} gt={len(gt_boxes)}"
        panels.append((title, _bgr_to_rgb(annotated)))
    _panel_grid(panels, cols, out_path, suptitle="HOG+SVM detections (green) vs ground truth (red)")
    return out_path


# ---------------------------------------------------------------------------
# 2. False positives / 3. Missed pedestrians
# ---------------------------------------------------------------------------


def plot_false_positives(
    examples: list[tuple[str, list[Detection], list[Box]]],
    out_path: Path = config.PLOTS_DIR / "false_positives.png",
    max_examples: int = config.MAX_EXAMPLES_PER_PLOT,
    cols: int = 3,
) -> Path:
    """Highlight false-positive detections (orange) alongside GT (red)."""
    panels: list[tuple[str, np.ndarray]] = []
    shown = 0
    for name, detections, gt_boxes in examples:
        from src.evaluation import match_detections

        matches, unmatched_preds, _ = match_detections(detections, gt_boxes)
        if not unmatched_preds:
            continue
        img = _downscale(_load_bgr(name))
        canvas = img.copy()
        for i in unmatched_preds:
            x, y, w, h = detections[i].box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), config.FP_COLOR, 2)
            cv2.putText(canvas, f"FP {detections[i].score:.2f}", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.FP_COLOR, 1, cv2.LINE_AA)
        for x, y, w, h in gt_boxes:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), config.GT_COLOR, 2)
        panels.append((f"{Path(name).name}  FP={len(unmatched_preds)}", _bgr_to_rgb(canvas)))
        shown += 1
        if shown >= max_examples:
            break
    _panel_grid(panels, cols, out_path, suptitle="False positives (orange) vs ground truth (red)")
    return out_path


def plot_missed_pedestrians(
    examples: list[tuple[str, list[Detection], list[Box]]],
    out_path: Path = config.PLOTS_DIR / "missed_pedestrians.png",
    max_examples: int = config.MAX_EXAMPLES_PER_PLOT,
    cols: int = 3,
) -> Path:
    """Highlight missed ground-truth pedestrians (magenta) vs predictions (green)."""
    panels: list[tuple[str, np.ndarray]] = []
    shown = 0
    for name, detections, gt_boxes in examples:
        from src.evaluation import match_detections

        matches, _, unmatched_gts = match_detections(detections, gt_boxes)
        if not unmatched_gts:
            continue
        img = _downscale(_load_bgr(name))
        canvas = annotate_image(img, detections, [], draw_scores=True)
        for gi in unmatched_gts:
            x, y, w, h = gt_boxes[gi]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), config.FN_COLOR, 2)
        panels.append(
            (f"{Path(name).name}  missed={len(unmatched_gts)}", _bgr_to_rgb(canvas))
        )
        shown += 1
        if shown >= max_examples:
            break
    _panel_grid(
        panels, cols, out_path,
        suptitle="Missed pedestrians (magenta); green = detections",
    )
    return out_path


# ---------------------------------------------------------------------------
# 4. Educational HOG visualization
# ---------------------------------------------------------------------------


def plot_hog_visualization(
    image_path: Path,
    detector: "PedestrianDetector" = None,  # noqa: F821 - provided by caller
    out_path: Path = config.PLOTS_DIR / "hog_visualization.png",
) -> Path:
    """Illustrate the HOG representation for one sample image.

    Panels: original image -> grayscale with gradient magnitude -> HOG
    descriptor visualisation (per-cell gradient-orientation roses, drawn with
    skimage-free NumPy/OpenCV code) -> the same rose view resized to the
    64x128 detection window the SVM actually scores.
    """
    from src.detector import PedestrianDetector

    detector = detector or PedestrianDetector()
    img = _load_bgr(image_path)
    # Crop/pad to the 64x128 aspect around the centre so the HOG cells align
    # with a single detection window.
    target_w, target_h = 64, 128
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h) if w and h else 1.0
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = max(0, (target_h - resized.shape[0]) // 2)
    x_off = max(0, (target_w - resized.shape[1]) // 2)
    canvas[max(0, y_off):max(0, y_off) + resized.shape[0],
           max(0, x_off):max(0, x_off) + resized.shape[1]] = resized[:target_h, :target_w]

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 9 orientation bins as in Dalal-Triggs (unsigned gradients, 0..180 deg).
    n_bins = 9
    cell = 8
    cells_y, cells_x = target_h // cell, target_w // cell
    hog_img = np.zeros((cells_y * 20, cells_x * 20), dtype=np.float32)
    bin_idx = np.clip((ang // (180.0 / n_bins)).astype(int), 0, n_bins - 1)
    for cy in range(cells_y):
        for cx in range(cells_x):
            ys, xs = slice(cy * cell, (cy + 1) * cell), slice(cx * cell, (cx + 1) * cell)
            for b in range(n_bins):
                strength = float(mag[ys, xs][bin_idx[ys, xs] == b].sum())
                if strength <= 0:
                    continue
                theta = np.deg2rad((b + 0.5) * (180.0 / n_bins))
                # Fold the orientation onto the full circle for drawing.
                dx, dy = np.cos(theta), -np.sin(theta)
                length = min(20.0, strength / 1500.0)
                cx_px, cy_px = (cx + 0.5) * 20, (cy + 0.5) * 20
                cv2.line(
                    hog_img,
                    (int(cx_px - dx * length), int(cy_px - dy * length)),
                    (int(cx_px + dx * length), int(cy_px + dy * length)),
                    255.0, 1,
                )

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), dpi=config.PLOT_DPI)
    axes[0].imshow(_bgr_to_rgb(cv2.resize(img, (target_w * 2, target_h * 2))))
    axes[0].set_title("Original image")
    axes[1].imshow(cv2.addWeighted(gray.astype(np.float32), 0.55, mag, 0.45, 0), cmap="magma")
    axes[1].set_title("Grayscale + gradient magnitude")
    axes[2].imshow(hog_img, cmap="viridis")
    axes[2].set_title("HOG gradient orientations\n(9 bins x 8x8 cells)")
    # Show the actual descriptor statistics the SVM consumes.
    descriptor = np.asarray(detector.hog.compute(gray), dtype=np.float32).reshape(-1)
    axes[3].hist(descriptor, bins=60, color="steelblue")
    axes[3].set_title(f"HOG descriptor values\n({descriptor.size} dims fed to the SVM)")
    axes[3].set_xlabel("descriptor value")
    for ax in axes[:3]:
        ax.axis("off")
    fig.suptitle("How the HOG + SVM detector sees a pedestrian", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
