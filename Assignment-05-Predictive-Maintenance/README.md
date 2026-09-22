# Assignment 5 – Predictive Maintenance from Sensor Logs

## Objective

Official assignment: **"Predictive Maintenance from Sensor Logs"** —
*"Classify component failures using sensor data and Python ML models."*

This project builds an end-to-end, leakage-safe classification pipeline that
predicts machine failure from CNC-mill sensor readings, compares three
imbalance-aware models, and saves the winner behind a prediction CLI.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning
Repository, dataset id **601** (CC BY 4.0; Matzka, 2020). The synthetic
ground truth from the AI4I 2020 challenge: 10,000 operating points of a CNC
milling machine.

- Local copy: `data/ai4i2020.csv` (~0.5 MB, byte-identical; provenance in
  `data/README.md`)
- `src/data_loader.py` re-downloads the official UCI zip automatically if
  the file is missing, so the pipeline works from a clean clone.
- **Shape used: 10,000 rows × 14 columns**, of which 5 are target-derived
  failure-mode flags (excluded as leakage) and 2 are identifiers (dropped).
- **Target: `Machine failure`** (binary).

Measured class distribution — **strong imbalance (28.5 : 1)**:

| Class | Rows | Share |
|---|---:|---:|
| No failure (0) | 9,661 | 96.61% |
| Failure (1) | 339 | 3.39% |

Data quality (measured): **0 missing cells, 0 exact duplicate rows**, clean
dtypes. The five failure-mode flags are highly imbalanced too:
TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.

## Problem

Predictive maintenance trains models on live sensor readings to predict a
failure *before* it happens, so maintenance can be scheduled proactively
instead of reacting to breakdowns. The economics are asymmetric: a missed
failure (false negative) can destroy a tool or halt production, while a
false alarm merely triggers an inspection. With failures at only 3.39% of
samples, a trivial "never fail" classifier scores 96.6% accuracy while
catching zero failures — so accuracy alone is meaningless here, and the
failure class is the explicit positive class throughout.

## Data Preparation

- **Validation**: `data_loader.validate()` enforces the expected 14-column
  schema and a strictly binary target; loading fails loudly otherwise.
- **Missing values / duplicates**: none exist in this dataset (0 and 0
  measured); exact-duplicate removal still runs defensively. No rows were
  fabricated or dropped.
- **Leakage exclusion (the critical step)**: the dataset ships five
  per-failure-mode indicator columns — `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
  (tool wear, heat dissipation, power, overstrain, random failures). These
  are the *components* of the target and are only knowable after a failure
  has been diagnosed; using them as inputs would let the model "cheat"
  (most flagged rows are labelled failures by construction). They are
  excluded from the feature matrix, along with `UDI` (row number) and
  `Product ID` (10,000 unique serials whose useful part — quality variant —
  is already in `Type`).
- **Feature engineering** (row-wise, target-independent, all documented):

| Feature | Formula | Why it is useful |
|---|---|---|
| `temp_diff_k` | process temp − air temp | The machine maintains a temperature differential; a collapsing differential signals heat-dissipation problems. |
| `power_w` | torque × 2π × rpm / 60 | Mechanical power in watts; abnormal power stresses the spindle (power-failure mode). |
| `torque_x_wear` | torque × tool wear | Cutting force applied through a worn tool; tool-overload wear failure is driven by this combination. |

  Final feature set (9): 5 raw sensor/operational columns + 3 engineered +
  `Type` (product quality L/M/H).
- **Encoding / scaling**: numeric → median-impute → `StandardScaler`;
  `Type` → most-frequent-impute → `OneHotEncoder(handle_unknown="ignore")`,
  all inside a `ColumnTransformer`.
- **Leakage-safe by construction**: the transformer learns only inside
  `model.fit(X_train, y_train)`; nothing is fitted before the stratified
  80/20 split; the feature matrix is built by explicit column selection, so
  the target physically cannot enter the inputs. Tests enforce all of this.

## Class Imbalance

Accuracy is reported but never used for selection: at 3.39% positives it is
dominated by the majority class. Instead:

- **class_weight="balanced"** on all three models (inversely-frequency
  class weights). A simple class-weighted approach was preferred over SMOTE
  — it is one parameter, applied inside `fit` on training data only, adds no
  dependency, and gives a solid result here.
- **Threshold-independent metrics**: ROC-AUC and PR-AUC (average precision),
  the latter being far more informative than ROC-AUC under this imbalance.
- **Failure-class precision / recall / F1** at a fixed 0.5 decision
  threshold (`zero_division=0` used throughout).

## Models

Three CPU-friendly classifiers (seed 42, sensible fixed parameters, no
hyperparameter tuning), each wrapped with the shared preprocessor in a
sklearn `Pipeline`:

| Model | Key parameters |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=2000` |
| Random Forest | `n_estimators=300`, `class_weight="balanced"`, `n_jobs=-1` |
| Hist Gradient Boosting | `max_iter=200`, `lr=0.1`, `class_weight="balanced"` |

## Model Selection Criterion

**Fixed in `config.py` before any model was trained:** primary metric =
failure-class **F1** on the held-out test set at threshold 0.5; tie-breakers
= PR-AUC, then failure recall. Applied verbatim after training.

## Evaluation

Stratified 80/20 split (seed 42) → 8,000 train (271 failures) / 2,000 test
(68 failures). All numbers below are **actual measured test-set results**
(failure class = positive, threshold 0.5):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8595 | 0.1802 | 0.8824 | 0.2993 | 0.9384 | 0.4295 |
| Random Forest | 0.9910 | 0.9630 | 0.7647 | 0.8525 | 0.9780 | 0.8717 |
| **Hist Gradient Boosting** | **0.9915** | 0.9180 | 0.8235 | **0.8682** | 0.9767 | **0.8898** |

Confusion matrix of the selected model (2,000 test rows):
**TP 56 · FP 5 · FN 12 · TN 1,927**.

Reading the table honestly: Logistic Regression achieves the highest recall
(0.8824) but at terrible precision (0.18 — 273 false alarms) because a
linear boundary cannot capture the failure rules; the two tree ensembles
dominate. Random Forest has the best precision (0.9630) and ROC-AUC (0.9780)
but misses more failures; **Hist Gradient Boosting wins the predefined F1
criterion (0.8682) and PR-AUC (0.8898)** and is the selected model. Note
that all three models have high accuracy — which is exactly why accuracy was
not used to choose between them.

## Error Analysis

For the selected Hist Gradient Boosting model on the test set:

- **False negatives: 12** (missed failures; failure recall 0.8235). Their
  predicted probabilities are very low (mean 0.033, max 0.213) — the model
  is confidently wrong, not borderline. Their profile: 10 of the 12 have
  moderate-to-high tool wear (91–246 min) but *unremarkable* torque/power —
  failures whose sensor signature does not match the strong torque-and-wear
  pattern the model learned. This is the practical cost of precision: the
  rare "quiet" failure slips through.
- **False positives: 5** (false alarms). Cheap by comparison — an
  unnecessary inspection, not a breakdown.
- Example false negatives (sensor values, first rows of
  `artifacts/metrics.json` error analysis; e.g. 1465 rpm, 59.1 Nm, 91 min
  wear, predicted probability 0.213) are printed by `python -m src.train`.

## Visualizations

Generated under `artifacts/plots/`:

| File | Content |
|---|---|
| `class_distribution.png` | Target distribution with counts and percentages |
| `confusion_matrix.png` | Selected model, test set |
| `roc_curve.png` | ROC curves for all three models with AUC |
| `precision_recall_curve.png` | PR curves with average precision and chance line |
| `model_comparison.png` | All six metrics side by side (best per metric highlighted) |

## Project Structure

```
Assignment-05-Predictive-Maintenance/
├── README.md, requirements.txt, pytest.ini, conftest.py, .gitignore
├── data/
│   ├── README.md                # UCI provenance + auto-download note
│   └── ai4i2020.csv             # official dataset copy (~0.5 MB)
├── src/
│   ├── config.py                # paths, seed 42, schema, selection criterion
│   ├── data_loader.py           # load/validate/dedup + UCI fallback
│   ├── feature_engineering.py   # 3 documented physical features
│   ├── preprocessing.py         # leakage-safe ColumnTransformer
│   ├── models.py                # 3 class-weighted classifiers
│   ├── evaluation.py            # imbalance-aware metrics, plots, error analysis
│   ├── train.py                 # end-to-end orchestration
│   └── predict.py               # demo + CSV prediction CLI
├── tests/                       # 36 tests across 4 files
├── models/                      # joblib (gitignored, regenerable) + metadata
└── artifacts/                   # metrics.json, model_comparison.csv, plots/
```

`models/*.joblib` (680 KB here) is excluded via `.gitignore` to keep the
repository GitHub-friendly; it is regenerated deterministically by
`python -m src.train`. `model_metadata.json` (selected model, features,
threshold, seed) is kept.

## How to Run

```bash
cd Assignment-05-Predictive-Maintenance
python -m pip install -r requirements.txt
python -m src.train     # trains all models, saves artifacts + best model
python -m src.predict   # demo predictions from the saved model
python -m src.predict --csv sensors.csv   # batch mode (raw sensor columns)
python -m pytest tests -v
```

Demo output from the actual saved model (no hardcoded values):

```
Machine 1: Type=L speed=1500 rpm, torque=38.5 Nm, wear=15 min
      -> NO FAILURE  (failure probability: 0.000)
Machine 2: Type=L speed=1300 rpm, torque=65.0 Nm, wear=220 min
      -> FAILURE  (failure probability: 1.000)
Machine 3: Type=M speed=2800 rpm, torque=15.0 Nm, wear=40 min
      -> NO FAILURE  (failure probability: 0.001)
```

Training runs in ~30 s on a normal laptop CPU; the full test suite takes
~4 s.

## Testing

`python -m pytest tests -v` → **36 passed** in ~4 s. Coverage: dataset
schema/binary target/documented class counts; failure-mode flags verified as
target components; engineered-feature math (including a physical check that
38.5 Nm at 1500 rpm = 6,048 W) and target-independence; leakage columns
absent from the feature matrix and transformer; imputation/scaling/one-hot
behaviour incl. unknown categories and frozen fitted statistics; all three
models train on a subset with correct probability shapes; the selection
function honours the predefined tie-breaker order; model save/reload
parity; both prediction-CLI modes.

## Conclusion

On the real AI4I 2020 data, a class-weighted Hist Gradient Boosting
classifier detects **82.4% of machine failures with 91.8% precision**
(F1 0.8682, PR-AUC 0.8898) from nine sensor features — versus a 3.39%
baseline prevalence, i.e. the model ranks failures ~26× better than chance.
The experiment also demonstrates *why* imbalance-aware evaluation matters:
the linear baseline posts 86% accuracy while flagging 273 false alarms, and
"never fail" would score 96.6% accuracy while finding nothing. The remaining
weakness is measured, not hidden: 12 quiet failures with atypical sensor
signatures are missed. This is an educational classification exercise on a
public synthetic dataset — not a production maintenance system.
