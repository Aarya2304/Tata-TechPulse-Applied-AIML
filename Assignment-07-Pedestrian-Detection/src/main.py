"""Assignment 7 CLI: single-image detection, qualitative demo, full evaluation.

Usage
-----
    python -m src.main --image path/to/image.jpg   # detect pedestrians in one image
    python -m src.main --demo                      # qualitative examples from the dataset
    python -m src.main --evaluate                  # full quantitative evaluation
    python -m src.main --evaluate --limit 60       # evaluate a smaller subset
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2

from src import config
from src.data_loader import GroundTruthImage, ensure_dataset, list_negative_images, load_ground_truth
from src.detector import Box, Detection, PedestrianDetector, annotate_image
from src.evaluation import (
    DetectionMetrics,
    evaluate_image,
    save_metrics,
    save_per_image_csv,
)
from src.visualization import (
    plot_detection_examples,
    plot_false_positives,
    plot_hog_visualization,
    plot_missed_pedestrians,
)


def _save_annotated(image, detections, gt_boxes, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = annotate_image(image, detections, gt_boxes, draw_scores=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


# ---------------------------------------------------------------------------
# Single-image mode
# ---------------------------------------------------------------------------
def run_single_image(image_path: Path, out_path: Path | None = None) -> list[Detection]:
    """Detect pedestrians in one image and save an annotated copy."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    detector = PedestrianDetector()
    t0 = time.time()
    detections = detector.detect(image)
    elapsed = time.time() - t0
    out_path = out_path or (config.ANNOTATED_DIR / f"detected_{Path(image_path).name}")
    _save_annotated(image, detections, [], out_path)
    print(f"Image: {image_path}")
    print(f"Detected pedestrians: {len(detections)}  ({elapsed:.2f}s)")
    for det in detections:
        x, y, w, h = det.box
        print(f"  box=(x={x}, y={y}, w={w}, h={h})  confidence={det.score:.3f}")
    print(f"Annotated image saved to: {out_path}")
    return detections


# ---------------------------------------------------------------------------
# Demo mode (qualitative examples on real dataset images)
# ---------------------------------------------------------------------------
def run_demo(detector: PedestrianDetector | None = None, limit: int = 4) -> None:
    """Run the detector on a handful of annotated images and save the outputs."""
    detector = detector or PedestrianDetector()
    gt_images = load_ground_truth()
    for gt in gt_images[:limit]:
        image = cv2.imread(str(gt.image_path))
        detections = detector.detect(image)
        out_path = config.ANNOTATED_DIR / f"detected_{gt.image_path.name}"
        _save_annotated(image, detections, gt.boxes, out_path)
        print(f"{gt.image_path.name}: {len(detections)} detections / {gt.n_persons} GT persons "
              f"-> {out_path}")


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------
def run_evaluation(limit: int | None = None, make_plots: bool = True) -> DetectionMetrics:
    """Evaluate the HOG+SVM detector against INRIA ground truth."""
    ensure_dataset()
    detector = PedestrianDetector()
    gt_images = load_ground_truth()
    rng = random.Random(config.SUBSET_SEED)
    if limit and limit < len(gt_images):
        gt_images = rng.sample(gt_images, limit)
        gt_images.sort(key=lambda g: g.image_path.name)
    print(f"Evaluating {len(gt_images)} annotated images "
          f"(IoU>={config.IOU_THRESHOLD}, NMS t={config.NMS_THRESHOLD}, "
          f"conf>{config.CONFIDENCE_THRESHOLD})...")

    metrics = DetectionMetrics()
    examples_cache: list[tuple[str, list[Detection], list[Box]]] = []
    neg_examples: list[tuple[str, list[Detection], list[Box]]] = []
    t_start = time.time()

    neg_files = list_negative_images()
    neg_sample = rng.sample(neg_files, min(len(neg_files), 20)) if neg_files else []

    for i, gt in enumerate(gt_images):
        image = cv2.imread(str(gt.image_path))
        if image is None:
            continue
        detections = detector.detect(image)
        result = evaluate_image(detections, gt.boxes, gt.image_path.name)
        metrics.add_image_result(result)
        if len(examples_cache) < 60 and (i < 9 or result.false_positives or result.false_negatives):
            examples_cache.append((str(gt.image_path), detections, gt.boxes))
        if i < 30:  # save annotated outputs for the first images
            _save_annotated(image, detections, gt.boxes,
                            config.ANNOTATED_DIR / f"detected_{gt.image_path.name}")
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(gt_images)}] running P={metrics.precision:.3f} "
                  f"R={metrics.recall:.3f}")

    # Pedestrian-free images: any detection there is a false positive.
    for neg_path in neg_sample:
        image = cv2.imread(str(neg_path))
        if image is None:
            continue
        detections = detector.detect(image)
        if detections:
            neg_examples.append((str(neg_path), detections, []))
        metrics.add_image_result(
            evaluate_image(detections, [], neg_path.name)
        )

    elapsed = time.time() - t_start
    summary = save_metrics(metrics)
    save_per_image_csv(metrics)

    runtime = {
        "evaluated_images": metrics.images_evaluated,
        "total_runtime_seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(1, metrics.images_evaluated), 3),
        "machine_cpu": "Intel-based laptop CPU (x64, Windows)",
        "opencv_version": cv2.__version__,
    }
    config.RUNTIME_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(config.RUNTIME_JSON, "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Runtime: {elapsed:.1f}s total, {runtime['seconds_per_image']}s/image "
          f"(OpenCV {cv2.__version__})")

    if make_plots:
        plot_detection_examples(examples_cache[:6])
        plot_false_positives(examples_cache + neg_examples)
        plot_missed_pedestrians(examples_cache)
        plot_hog_visualization(gt_images[0].image_path, detector)
        print("Plots written to", config.PLOTS_DIR)
    return metrics


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="HOG+SVM pedestrian detection (OpenCV)")
    parser.add_argument("--image", type=Path, help="run detection on a single image")
    parser.add_argument("--demo", action="store_true",
                        help="run qualitative demo on annotated dataset images")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the full quantitative evaluation")
    parser.add_argument("--limit", type=int, default=config.EVAL_LIMIT,
                        help="number of annotated images to evaluate")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plot generation during evaluation")
    args = parser.parse_args()

    for directory in (config.PLOTS_DIR, config.ANNOTATED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if args.image:
        run_single_image(args.image)
    elif args.demo:
        run_demo()
    elif args.evaluate:
        run_evaluation(limit=args.limit, make_plots=not args.no_plots)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
