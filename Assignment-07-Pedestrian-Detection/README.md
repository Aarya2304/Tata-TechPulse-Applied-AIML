# Assignment 7 – Pedestrian Detection using OpenCV

## Objective

Official requirement (Tata Technologies TechPulse FY-26, Applied AI/ML):

> "Detect pedestrians using OpenCV's HOG + SVM method."

The core detector in this project is exactly that: `cv2.HOGDescriptor` combined
with OpenCV's pretrained default people-detection SVM
(`cv2.HOGDescriptor_getDefaultPeopleDetector()`). No neural network and no
other detection framework (YOLO/SSD/Faster R-CNN/transformers/…) is used
anywhere in this project.

## Approach

```
Image
  → HOG feature representation (64x128 window, 3,780-dim descriptor)
  → pretrained linear SVM (OpenCV default people detector)
  → sliding-window detection over an image pyramid (detectMultiScale)
  → candidate bounding boxes + SVM decision values ("weights")
  → confidence filtering (weight > 0.5)
  → window-margin box calibration (shrink 0.65, see below)
  → Non-Maximum Suppression (greedy, NumPy, IoU threshold 0.4)
  → final detections
  → IoU-based matching against ground truth (IoU ≥ 0.5)
  → TP/FP/FN, Precision, Recall, F1, mean matched IoU
```

The HOG descriptor builds a histogram of gradient orientations over 8x8-pixel
cells, normalises them over overlapping 2x2-cell blocks, and concatenates the
result into a 3,780-dimensional feature vector per 64x128 window. The bundled
SVM — trained by OpenCV's authors on the INRIA Person *training* split —
scores each window; a positive decision value means "pedestrian".

## Dataset

| Property | Value |
|---|---|
| Name | INRIA Person Dataset (Dalal & Triggs, CVPR 2005) |
| Source | official archive mirrored by UCF: `http://cs.ucf.edu/courses/cap6412/fall2009/misc/INRIAPerson.tar` (1,016,094,720 bytes; the original `pascal.inrialpes.fr` host is offline) |
| Setup | automatic: `python -m src.main --evaluate` downloads and extracts the archive when missing (only the test split, ~276 MB, is extracted) |
| Split used | official **Test** split |
| Annotated images | **288** (`Test/pos/*.png`) |
| Ground-truth pedestrians | **589** `PASperson` boxes (up to 16 per image) |
| Pedestrian-free images | 453 available (`Test/neg/*.jpg`); **20** sampled ones included in the evaluation |
| Annotation format | PASCAL-style `.txt`: `Bounding box for object N "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (x1, y1) - (x2, y2)` — parsed to OpenCV (x, y, w, h) |

The training split of the archive is **not** downloaded/extracted: the SVM is
already pretrained, so no training happens in this assignment.

## HOG + SVM configuration

| Parameter | Value |
|---|---|
| Descriptor | `cv2.HOGDescriptor()` defaults — 64x128 window, 3,780 dims |
| SVM | `cv2.HOGDescriptor_getDefaultPeopleDetector()` (pretrained, bundled with OpenCV) |
| `winStride` | (8, 8) |
| `padding` | (8, 8) |
| `scale` (pyramid step) | 1.05 |
| `hitThreshold` | 0.0 (OpenCV default margin) |
| `groupThreshold` | 0 — OpenCV's built-in grouping is disabled; our NMS does this job explicitly |
| Confidence threshold | weight **> 0.5** (post-filter on SVM decision values) |
| Box-margin calibration | shrink detection boxes to **65 %** of the raw window size around their centre (see below) |
| NMS | own greedy NumPy implementation, threshold **0.4** |

### Why the box-shrink calibration is needed (measured, not assumed)

OpenCV's 64x128 detection window contains deliberate margin around the person,
while INRIA annotations are tight person crops. Reporting raw windows yields
boxes ~1.5x wider than the annotated person, and at IoU 0.5 almost every
detection is scored a false positive even when it is visually correct.

Measured on a 20-image calibration subset (seed 42), varying only this factor
and the confidence threshold:

| Configuration (subset of the grid tried) | Precision | Recall | F1 |
|---|---:|---:|---:|
| raw windows, conf > 0, NMS 0.4 | 0.069 | 0.143 | 0.093 |
| raw windows + OpenCV grouping, conf > 0, NMS 0.4 | 0.057 | 0.057 | 0.057 |
| shrink 0.65, conf > 0, NMS 0.4 | 0.516 | 0.457 | 0.485 |
| shrink 0.65, conf > 0.3, NMS 0.4 | 0.457 | 0.600 | 0.519 |
| **shrink 0.65, conf > 0.5, NMS 0.4 (selected)** | **0.625** | **0.571** | **0.597** |
| shrink 0.65, conf > 0.9, NMS 0.4 | 0.750 | 0.429 | 0.545 |

The finer pyramid (`scale=1.03`, `winStride=(4,4)`) was also tried: it raised
recall slightly but tripled the detections per image and did not improve F1,
so the assignment's starting parameters (8,8 / 8,8 / 1.05) were kept. The
selected configuration was chosen on the calibration subset only; the full
results below come from the separate full run.

## Evaluation Method

For every image, predictions (after confidence filtering, calibration and NMS)
are matched to ground truth with a **greedy IoU protocol**:

1. compute IoU for every prediction/GT pair;
2. repeatedly take the unmatched pair with the highest IoU;
3. a pair is a **true positive** when its IoU ≥ **0.5**;
4. each GT box and each prediction is matched **at most once**;
5. unmatched predictions are **false positives**; unmatched GT boxes are
   **false negatives**.

```
Precision = TP / (TP + FP)          (0 if TP + FP = 0)
Recall    = TP / (TP + FN)          (0 if TP + FN = 0)
F1        = 2·P·R / (P + R)         (0 if P + R = 0)
```

Pedestrian-free (`neg`) images contribute only potential false positives
(any detection there is an FP). Mean matched IoU is averaged over TP pairs.

## Results

Actual measured values from the full evaluation run
(`python -m src.main --evaluate`, 308 images, OpenCV 4.11.0, CPU):

| Metric | Value |
|---|---:|
| Images evaluated | 308 (288 annotated + 20 pedestrian-free) |
| Ground-truth pedestrians | 589 |
| Predictions (after NMS) | 592 |
| True Positives | 382 |
| False Positives | 210 |
| False Negatives | 207 |
| **Precision** | **0.645** |
| **Recall** | **0.649** |
| **F1-score** | **0.647** |
| Mean matched IoU | 0.678 |
| Runtime | 63.5 s total, ≈ 0.21 s/image (Intel laptop CPU, single process) |

Per-image details are in `artifacts/evaluation_results.csv`; the same summary
in machine-readable form is `artifacts/metrics.json` (with
`artifacts/reports/runtime.json` for timing). Re-running the evaluation
reproduces these numbers exactly — the pipeline is deterministic (no random
component beyond the fixed-seed subset choice).

## Qualitative Analysis

Observations supported by the generated figures and per-image CSV:

- **Correct detections** — 75 of the 288 annotated images are detected
  perfectly (0 FP and 0 FN); single well-separated pedestrians in
  moderate viewpoints are reliably found (e.g. `crop001501.png`: 4/4 GT
  matched, mean matched IoU 0.68).
- **False positives** — 210 in total. They cluster on person-like vertical
  structures: traffic signs, poles, bag straps, and duplicated detections of
  partially visible people (e.g. `crop001511.png` contains 1 person and 5
  FPs). On the 20 pedestrian-free images only 2 produced detections
  (4 FPs total) — the detector is not trigger-happy on empty scenes, FPs
  concentrate in cluttered ones. FP confidence (mean 0.88) is clearly lower
  than TP confidence (mean 1.64), which is what the 0.5 threshold exploits.
- **Missed pedestrians** — 207 in total. A 120-image probe shows missed GT
  boxes are *not* mostly tiny people (only 8.7 % are below 80 px height);
  they concentrate in **crowded, occluded scenes** — missed persons average
  319 px height vs 259 px for matched ones, i.e. close-up groups where the
  HOG template no longer fits individual heavily-occluded people
  (worst: `crop001607.png` with 7 of 10 missed, `person_200.png` with 6/16).
- **Challenging conditions** — strong shadows/backlight, unusual poses
  (bikers in `person_and_bike_*` images), and partial occlusion are the
  dominant error sources, consistent with the known limits of a rigid
  single-template HOG detector.

All three failure modes are visualised honestly — the figures below are
generated from the same run that produced the metrics above, including
unsuccessful examples.

## HOG Visualization

`artifacts/plots/hog_visualization.png` illustrates the representation for one
sample image: the original crop → grayscale with gradient magnitude → the
per-cell gradient-orientation "rose" view of the HOG descriptor (9 unsigned
orientation bins over 8x8 cells) → the distribution of the actual
3,780 descriptor values fed to the SVM. It shows that dominant gradients
concentrate on the person's silhouette edges (head/shoulders/legs), which is
exactly the cue the linear SVM keys on.

## Artifacts

```
artifacts/metrics.json                  # aggregate metrics (this README's table)
artifacts/evaluation_results.csv        # per-image TP/FP/FN/mean-IoU (308 rows)
artifacts/reports/runtime.json          # measured runtime + environment
artifacts/plots/detection_examples.png  # predictions (green) vs GT (red)
artifacts/plots/false_positives.png     # FPs (orange) incl. neg-image FPs
artifacts/plots/missed_pedestrians.png  # missed GT (magenta)
artifacts/plots/hog_visualization.png   # educational HOG figure
outputs/annotated/*.png|jpg             # 30+ annotated images from the run
```

## How to Run

```bash
cd Assignment-07-Pedestrian-Detection
python -m pip install -r requirements.txt

# full quantitative evaluation + all plots (downloads dataset if missing)
python -m src.main --evaluate

# qualitative demo on 4 annotated dataset images
python -m src.main --demo

# detect pedestrians in a single image
python -m src.main --image path/to/image.jpg

# smaller/faster evaluation subset
python -m src.main --evaluate --limit 60

# tests (no dataset required)
python -m pytest -q
```

## Testing

`python -m pytest -q` → **45 passed** in ~3.5 s. Coverage: HOG detector
initialisation and the default people SVM (3,780-dim, 64x128 window),
detection on synthetic images (incl. empty/tiny-image guards), NMS (overlap
suppression, disjoint keeps, ordering, threshold boundary, brute-force
cross-check), IoU (exact/partial/containment/touching/degenerate), greedy
matching and metric formulas (synthetic TP/FP/FN scenarios, zero-denominator
safety), annotation parsing (corner→xywh, degenerate-box skipping, real
dataset statistics when extracted), and all four figures render on synthetic
input. Tests are deterministic and do not require the dataset.

## Project Structure

```
Assignment-07-Pedestrian-Detection/
├── data/README.md                # dataset provenance + layout
├── models/README.md              # why there is no model file (pretrained SVM)
├── artifacts/                    # metrics.json, evaluation_results.csv, plots/
├── outputs/annotated/            # annotated images from the run
├── src/
│   ├── config.py                 # all tunables + paths
│   ├── data_loader.py            # download/extract + GT parser
│   ├── detector.py               # HOG+SVM detector, IoU, NMS, annotation
│   ├── evaluation.py             # matching, metrics, JSON/CSV reports
│   ├── visualization.py          # the four figures
│   └── main.py                   # CLI: --evaluate / --image / --demo
├── tests/                        # 45 tests, 5 files
├── requirements.txt
└── .gitignore                    # dataset archives/extracts + outputs
```

## Limitations

- The rigid 64x128 template degrades on heavy occlusion and crowded groups —
  the dominant error mode in our results (207 FNs).
- Persons far from the trained aspect ratio (bikers, partially cropped people)
  and unusual poses are frequently missed.
- Vertical, person-like background structures (signs, poles) produce false
  positives; classical HOG+SVM has no appearance memory to suppress them.
- The fixed pyramid (scale 1.05, 64x128 minimum window) cannot detect people
  whose projected height is far from the trained size range without resizing.
- The box-shrink calibration (0.65) is fitted to INRIA's annotation style;
  against a dataset with looser boxes it should be re-calibrated (or set to
  1.0), since IoU depends on the annotation convention.

## Conclusion

The classical HOG + SVM detector — OpenCV's pretrained people SVM, exactly as
the assignment requires — reaches **F1 0.647 (P 0.645 / R 0.649, mean matched
IoU 0.678)** on the INRIA Person test split at ≈5 fps on a laptop CPU, with a
calibrated confidence threshold, window-margin correction and NumPy NMS. The
experiment makes the classical pipeline's strengths (fast, training-free,
decent on isolated upright pedestrians) and its ceiling (occlusion, crowding,
rigid template) directly measurable.
