# Assignment 6 – Traffic Sign Classification using CNN

## Objective

Official assignment: **"Traffic Sign Classification using CNN"** —
*"Train a CNN to recognize traffic signs using the GTSRB dataset."*

This project trains a compact TensorFlow/Keras CNN from scratch on the
German Traffic Sign Recognition Benchmark (43 classes), with a leakage-safe
train/val/test design, light orientation-preserving augmentation,
imbalance-aware class weights, full evaluation beyond accuracy, and a
prediction CLI.

## Dataset

**GTSRB** — Houben et al., IJCNN 2011. Official download host (benchmark
organisers' archive): `sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/`
(linked from <https://benchmark.ini.rub.de/gtsrb_dataset.html>).

- `python -m src.train` **auto-downloads and extracts** the three official
  zips (~350 MB download, ~570 MB extracted) into `data/downloads/` if
  missing; nothing large is committed (`data/downloads/` is gitignored).
  See `data/README.md`.
- **Classes used: 43** (all), integer labels 0–42 with the official
  class names (`src/config.py`, saved to `models/class_names.json`).

Measured counts from this run (nothing fabricated):

| Split | Images | Source |
|---|---:|---|
| Training pool | 39,209 | official `Final_Training` (43 folders: 210 – 2,250 images/class, 10.7× imbalance) |
| Validation | 5,881 (15%) | stratified carve-out from the training pool (seed 42) |
| Training (post-split) | 33,328 | 80% remainder, class weights computed on these only |
| Test | 12,630 | official `Final_Test` + official labels CSV, never seen in training |

Images are variable-size PPM (e.g. 20×24 up to 200×200+ in a small sample);
one test row was measured as 32×31. All images are resized to **32×32 RGB**
(`resize_with_pad`) and normalized `pixel / 255.0`. Discovery reported
**zero unreadable images**; every image in the official archives decoded.

## CNN Architecture

Implemented in `src/model.py` (`build_cnn()`, 332,427 parameters, ~4 MB):

```
Input (32,32,3)
→ [Conv2D 32 3x3 → BatchNorm → ReLU] ×2 → MaxPooling
→ [Conv2D 64 3x3 → BatchNorm → ReLU] ×2 → MaxPooling
→ [Conv2D 128 3x3 → BatchNorm → ReLU] ×2 → MaxPooling
→ GlobalAveragePooling2D
→ Dense 256 (ReLU) → Dropout 0.4
→ Dense 43 (Softmax)
```

Softmax output over 43 classes, `sparse_categorical_crossentropy` (integer
labels), Adam (lr 1e-3). Deliberately compact for CPU training.

## Training

- **Optimizer / loss**: Adam 1e-3 / sparse categorical cross-entropy
- **Batch size**: 64 · **Max epochs**: 30 (early stopping ran it to 29)
- **Augmentation (training split only)**: rotation ±~11°, translation ±6%,
  zoom ±8%, `fill_mode="nearest"` — created as Keras preprocessing layers
  inside the `tf.data` map. **No flips** (traffic-sign orientation is
  meaningful; a horizontal-flip layer is explicitly tested against).
- **Callbacks**: `ModelCheckpoint(monitor="val_accuracy", save_best_only)`
  → `models/best_traffic_sign_cnn.keras`; `EarlyStopping(monitor="val_loss",
  patience=5, restore_best_weights=True)`; `ReduceLROnPlateau(factor=0.5,
  patience=3)`.
- **Class weights**: sklearn `compute_class_weight("balanced")` computed
  from the **training split only** and passed to `fit`.
- **Seeds**: `random`, NumPy and TensorFlow all set to **42**. Runs are
  highly reproducible but **not guaranteed bit-for-bit identical** (oneDNN
  reduction order / backend kernels) — noted in `metrics.json`.
- **Memory**: images stay on disk and are decoded/resized/normalized lazily
  by `tf.data` with `prefetch(AUTOTUNE)`; the full-resolution dataset is
  never duplicated in RAM (peak well under 16 GB).
- **Runtime**: ~40 s/epoch, ~20 min total on a normal laptop CPU.

**Model selection**: the best checkpoint by **validation accuracy**
(`ModelCheckpoint`), which coincided with the best val_loss epoch
(val_accuracy 0.9995 at epoch 23 of 29; early stopping then restored the
best weights). No architecture search was performed, per the assignment.

## Evaluation

All values below are **actual measured results** on the official 12,630-image
test set (`artifacts/metrics.json`):

| Metric | Value |
|---|---:|
| Test Accuracy | 0.9793 |
| Macro Precision | 0.9631 |
| Macro Recall | 0.9654 |
| Macro F1 | 0.9630 |
| Weighted Precision | 0.9800 |
| Weighted Recall | 0.9793 |
| Weighted F1 | 0.9791 |

Incorrect predictions: **262 / 12,630 (2.07%)**.
Full per-class report: `artifacts/classification_report.csv`.

## Error Analysis

Total misclassified: 262. Measured confusion concentrates in visually
similar families:

| True → Predicted | Count | Meaning |
|---|---:|---|
| 21 → 18 | 24 | Double curve → General caution (triangular warnings) |
| 27 → 21 | 21 | Pedestrians → Double curve |
| 2 → 1 | 15 | 50 km/h → 30 km/h speed limit (red-circle family) |
| 30 → 23 | 15 | Beware of ice/snow → Slippery road |
| 14 → 15 | 15 | Stop → No vehicles |
| 6 → 42 | 12 | End of 80-limit → End of no passing (trucks) |
| 12 → 13 | 12 | Priority road → Yield |
| 38 → 40 | 11 | Keep right → Roundabout mandatory |

Weakest classes by F1 (all rare, ≤ 150 test samples):
**Double curve 0.686** (P 0.706 / R 0.667), **Pedestrians 0.692**
(recall 0.600 — 24 of its 60 test images missed), Roundabout mandatory
0.871, Beware of ice/snow 0.912, Slippery road 0.935. The pattern is
consistent: dark triangular warning signs are mutually confusable, and
strong classes (most speed limits ≥ 0.99 F1) mask them in plain accuracy.
`artifacts/plots/misclassified_examples.png` shows 12 misclassified test
images with true class, predicted class and confidence.

## Artifacts

| File | Content |
|---|---|
| `models/best_traffic_sign_cnn.keras` | trained CNN, best-val checkpoint (4.0 MB, gitignored; regenerate with `python -m src.train`) |
| `models/class_names.json` | class id → official sign name |
| `artifacts/metrics.json` | all headline metrics + run metadata |
| `artifacts/classification_report.csv` | per-class precision/recall/F1/support |
| `artifacts/plots/class_distribution.png` | 43-class training distribution (210–2,250) |
| `artifacts/plots/training_history.png` | accuracy & loss curves, train vs validation |
| `artifacts/plots/confusion_matrix.png` | 43×43 matrix: row-normalised colour + raw counts |
| `artifacts/plots/misclassified_examples.png` | misclassified images with true/pred/confidence |

## How to Run

```bash
cd Assignment-06-Traffic-Sign-Classification
python -m pip install -r requirements.txt
python -m src.train                            # downloads GTSRB if missing, trains, saves artifacts
python -m src.predict path/to/sign.png         # classify an image (ppm/png/jpg)
python -m src.predict                          # demo mode on official test images
pytest -q
```

Demo output from the actual saved model (no hardcoded values):

```
[OK ] 00000.ppm  true=16 (Vehicles over 3.5 metric tons ) -> pred=16  conf=1.000
[OK ] 00001.ppm  true= 1 (Speed limit (30km/h)) -> pred= 1  conf=1.000
[OK ] 00002.ppm  true=38 (Keep right) -> pred=38  conf=1.000
[OK ] 00003.ppm  true=33 (Turn right ahead) -> pred=33  conf=1.000
[OK ] 00004.ppm  true=11 (Right-of-way at the next inter) -> pred=11  conf=1.000
```

## Testing

`pytest -q` → **32 passed, 1 skipped** in ~10 s (the skip is the
class-mapping file check before the first training run). The suite is fully
offline: a synthetic GTSRB-style PPM tree and a tiny 1-epoch CNN provide the
fixtures. Coverage: label validation and out-of-range rejection; discovery
on a synthetic tree incl. deterministic ordering and missing-dir errors;
official test-GT parsing incl. bad-schema rejection; stratified split
determinism and class coverage; decode/resize of variable-size images to
32×32; normalization range; augmentation bounds and **absence of flip
layers**; dataset batch shapes; CNN input/output shapes, softmax sums,
architecture block counts and filter progression (32/64/128); parameter
budget (< 5 M); compile settings; callback configuration; save/load
round-trip; prediction structure with probabilities summing to 1; CLI modes
including the missing-model error path.

## Conclusion

A 332K-parameter CNN trained from scratch in ~20 minutes on CPU reaches
**97.93% test accuracy** (macro F1 0.9630) on the official GTSRB test set —
respectably close to much larger published baselines while remaining a
compact, understandable architecture. The honest headline is nuanced:
macro metrics sit ~1.6 points below accuracy because rare triangular
warning signs (Double curve, Pedestrians) are the residual confusion
clusters, quantified in the error analysis. This is an educational
image-classification exercise on a public benchmark dataset — not a
production traffic-sign recognition system.
