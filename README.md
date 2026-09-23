# Tata Technologies TechPulse — Applied AI/ML

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688)
![Docker](https://img.shields.io/badge/Docker-available-2496ED)

This repository contains the complete set of **10 assignments** from the Tata
Technologies **TechPulse FY-26 Applied AI/ML** program. Each assignment lives in
its own self-contained folder with source code, tests, documentation and
generated artifacts, covering the full arc from data preprocessing and
classical ML to deep learning, computer vision, NLP and MLOps.

---

## Technologies & Skills

### Machine Learning
- Regression (Linear, Ridge, Random Forest, Gradient Boosting)
- Classification (Logistic Regression, Random Forest, HistGradientBoosting)
- Class-imbalance handling (class weighting, imbalance-aware metrics)
- Genetic algorithms (neuroevolution for a simulated driving agent)
- Feature importance (impurity-based and permutation-based)

### Deep Learning
- CNN (traffic-sign classification)
- LSTM (sentiment classification)
- TensorFlow / Keras

### Computer Vision
- GTSRB traffic-sign recognition (43 classes)
- OpenCV `HOGDescriptor` + default people-detection SVM
- Sliding-window detection, non-maximum suppression, IoU-based evaluation

### NLP
- Text cleaning and tokenization (train-only vocabulary fitting)
- Sequence padding, embeddings
- Sentiment classification with LSTM

### MLOps
- MLflow experiment tracking (params, metrics, model, artifacts)
- Docker + Docker Compose packaging with health checks
- FastAPI inference service (Pydantic validation)
- GitHub Actions CI workflow
- Local deployment simulation (build → run → health → predict)

### Data Science
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Profiling, missing-value/outlier handling, scaling, encoding
- Leakage-aware preprocessing with `Pipeline` / `ColumnTransformer`

---

## Assignments Overview

| # | Assignment | Core Technique | Dataset / Data | Key Result | Tests |
|---|---|---|---|---|---|
| 1 | Car Mileage Estimation | Regression + Random Forest | Auto MPG | R² 0.864 | 28/28 |
| 2 | Simulated Driving Agent | Genetic algorithm + neural network | Procedural track (seed 42) | Fitness 108.55 → 9079.55 (8.04 laps) | 53/53 |
| 3 | Data Cleaning & Preprocessing | Imputation, IQR winsorization, scaling | CarDekho Car details v3 | Missing cells 1058 → 0 | 48/48 |
| 4 | Vehicle Price Prediction | Regression + model comparison | CarDekho Car details v3 | R² 0.9300 (Random Forest) | 35/35 |
| 5 | Predictive Maintenance | Imbalanced classification | AI4I 2020 | Failure-class F1 0.8682 | 36/36 |
| 6 | Traffic Sign Classification | CNN (TensorFlow/Keras) | GTSRB | Accuracy 0.9793 | 33/33 |
| 7 | Pedestrian Detection | OpenCV HOG + SVM | INRIA Person (test split) | F1 0.647 | 45/45 |
| 8 | Sentiment Analysis | LSTM (TensorFlow/Keras) | Amazon Automotive Reviews | F1 0.9473 | 30/30 |
| 9 | Feature Importance | Permutation vs impurity importance | CarDekho Car details v3 | R² 0.9722; `max_power_bhp` dominant | 21/21 |
| 10 | MLOps Workflow | MLflow + Docker + FastAPI + CI | Synthetic automotive (4,000 rows) | R² 0.8167; healthy containerized API | 32/32 |

---

## Assignments

### 1. Car Mileage Estimation
Predicts car mileage (MPG) from the classic UCI Auto MPG dataset with a
leakage-free sklearn pipeline, including horsepower imputation. Linear
Regression, Ridge and Random Forest are compared with 5-fold cross-validation;
Random Forest was the strongest evaluated model (MAE 1.941, R² 0.864).

- **Dataset**: Auto MPG (398 rows)
- **Techniques**: Linear/Ridge/Random Forest regression, cross-validation, MAE/RMSE/R²
- **Result**: Random Forest — MAE 1.941, RMSE 2.842, R² 0.864 (best of the three models)
- **Tests**: 28/28 passed
- [View Assignment 1](./Assignment-01-Car-Mileage/)

### 2. Simulated Driving Agent Behavior
Evolves a self-driving agent around a procedural closed-loop track using a
genetic algorithm over an 8-8-2 NumPy neural network driven by 7 ray sensors
plus normalized speed. Arc-length track progress is measured deterministically,
and the best agent is validated by replay on the same seeded track.

- **Dataset**: deterministic procedural track (668 m, seed 42)
- **Techniques**: GA (population 80, 50 generations, tournament selection, crossover, mutation, elitism), neuroevolution
- **Result**: best fitness 108.55 → 9079.55 across generations; final run 8.04 laps, all 900 trajectory points on-track
- **Tests**: 53/53 passed
- [View Assignment 2](./Assignment-02-Simulated-Driving-Agent/)

### 3. Data Cleaning & Preprocessing Techniques
Cleans the messy CarDekho automotive dataset end-to-end: numeric-string
parsing (mileage, engine, power, torque incl. kgm→Nm), duplicate handling,
domain validation, median imputation, IQR-based winsorization, and
Standard/Robust scaling — all wrapped in leakage-verified sklearn pipelines
with before/after reports and plots.

- **Dataset**: CarDekho "Car details v3" (8,128 × 13)
- **Techniques**: parsing, imputation, IQR winsorization, Standard/Robust scaling, ColumnTransformer
- **Result**: 8,128 → 6,907 rows after duplicate removal; missing cells 1,058 → 0; 876 values capped; processed matrix 6,907 × 54
- **Tests**: 48/48 passed
- [View Assignment 3](./Assignment-03-Data-Cleaning/)

### 4. Vehicle Price Prediction
Predicts used-vehicle selling prices from structured automotive data with
engineered features (vehicle age, km/year, brand, parsed torque/power) and a
leakage-safe `ColumnTransformer` pipeline. Four regressors are compared on a
log-transformed target with cross-validation, error analysis and feature
importance; Random Forest wins the predefined lowest-RMSE criterion.

- **Dataset**: CarDekho "Car details v3"
- **Techniques**: Linear/Ridge/Random Forest/Gradient Boosting, log1p target, CV, error analysis
- **Result**: Random Forest — MAE ₹71,951, RMSE ₹124,492, test R² 0.9300
- **Tests**: 35/35 passed
- [View Assignment 4](./Assignment-04-Vehicle-Price-Prediction/)

### 5. Predictive Maintenance from Sensor Logs
Classifies rare machine failures (3.39% failure rate) from the AI4I 2020
sensor dataset with leakage-aware features (`temp_diff_k`, `power_w`,
`torque_x_wear`) and class-weighted models. Accuracy alone is rejected in
favour of precision/recall/F1, ROC-AUC and PR-AUC; HistGradientBoosting is
selected by failure-class F1.

- **Dataset**: AI4I 2020 Predictive Maintenance (10,000 × 14)
- **Techniques**: class weighting, stratified split, imbalance-aware evaluation, error analysis
- **Result**: HistGradientBoosting — failure-class F1 0.8682, ROC-AUC 0.9767, PR-AUC 0.8898
- **Tests**: 36/36 passed
- [View Assignment 5](./Assignment-05-Predictive-Maintenance/)

### 6. Traffic Sign Classification using CNN
Trains a compact Keras CNN (332,427 parameters) on the official GTSRB dataset
(39,209 train / 12,630 test images, 43 classes) with lazy PPM decoding,
class weighting and flip-free augmentation. The best validation checkpoint is
evaluated on the untouched official test set.

- **Dataset**: GTSRB (43 classes)
- **Techniques**: CNN + BatchNorm + GAP + Dropout, augmentation, EarlyStopping/ReduceLROnPlateau, checkpoint selection
- **Result**: test accuracy 0.9793 (262/12,630 incorrect), macro F1 0.9630
- **Tests**: 33/33 passed
- [View Assignment 6](./Assignment-06-Traffic-Sign-Classification/)

### 7. Pedestrian Detection using OpenCV
Implements the classical detection pipeline with OpenCV's `HOGDescriptor` and
the default pretrained people-detection SVM, plus confidence filtering, NumPy
NMS and measured bounding-box calibration. Quantitatively evaluated against
real INRIA ground-truth boxes with IoU ≥ 0.5 matching.

- **Dataset**: INRIA Person test split (308 images, 589 ground-truth boxes)
- **Techniques**: HOG + SVM, sliding window, NMS, IoU matching, TP/FP/FN metrics
- **Result**: precision 0.645, recall 0.649, F1 0.647, mean matched IoU 0.678
- **Tests**: 45/45 passed
- [View Assignment 7](./Assignment-07-Pedestrian-Detection/)

### 8. Sentiment Analysis using LSTM
Classifies real vehicle-customer feedback (Amazon Automotive reviews) as
positive/negative using a Keras LSTM (803,137 parameters) over embeddings, with
a train-only-fitted tokenizer, pre-padding with masking, class weighting and a
stratified split. Includes confusion matrix, ROC/PR-AUC and error analysis.

- **Dataset**: Amazon Automotive Reviews (19,032 usable reviews)
- **Techniques**: Embedding → LSTM → Dropout → Dense → Sigmoid, train-only tokenizer, imbalance-aware metrics
- **Result**: accuracy 0.9028, F1 0.9473, ROC-AUC 0.8070, PR-AUC 0.9806
- **Tests**: 30/30 passed
- [View Assignment 8](./Assignment-08-Sentiment-LSTM/)

### 9. Feature Importance Visualization
Explains a RandomForestRegressor vehicle-price model with two complementary
importance methods — tree impurity importance and permutation importance on
the untouched test set — mapped correctly onto the one-hot-expanded feature
space and aggregated back to logical feature groups. Includes a
reproducibility check across seeds and clear non-causality interpretation.

- **Dataset**: CarDekho "Car details v3" (reused read-only)
- **Techniques**: impurity + permutation importance, one-hot grouping, comparison plots
- **Result**: MAE ₹67,110, RMSE ₹134,939, R² 0.9722; `max_power_bhp` strongest across both methods
- **Tests**: 21/21 passed
- [View Assignment 9](./Assignment-09-Feature-Importance/)

### 10. MLOps Workflow Simulation
Demonstrates the full ML lifecycle: deterministic synthetic automotive data →
RandomForestRegressor → MLflow tracking (params, metrics, model, artifacts) →
FastAPI prediction service → Docker image (model trained inside the image) →
health-checked local deployment simulation, gated by a GitHub Actions CI
workflow and a 32-test suite.

- **Dataset**: synthetic automotive data (4,000 rows, explicitly for MLOps demonstration)
- **Techniques**: MLflow, FastAPI, Docker/Docker Compose, GitHub Actions CI, health checks
- **Result**: MAE ₹125,252.34, RMSE ₹243,943.00, R² 0.8167; image built, container healthy, prediction verified from host
- **Tests**: 32/32 passed
- [View Assignment 10](./Assignment-10-MLOps-Workflow/)

---

## Selected Results

| # | Assignment | Headline Metric | Value |
|---|---|---|---:|
| 1 | Car Mileage Estimation | Random Forest R² | 0.864 |
| 4 | Vehicle Price Prediction | Random Forest R² | 0.9300 |
| 5 | Predictive Maintenance | HistGradientBoosting failure-class F1 | 0.8682 |
| 6 | Traffic Sign Classification | CNN test accuracy | 0.9793 |
| 7 | Pedestrian Detection | HOG + SVM F1 | 0.647 |
| 8 | Sentiment Analysis | LSTM F1 | 0.9473 |
| 9 | Feature Importance | Random Forest R² | 0.9722 |
| 10 | MLOps Workflow | Random Forest R² (synthetic data) | 0.8167 |

> These metrics come from **different datasets, targets and tasks** (MPG in
> miles per gallon, prices in INR, failure F1, image accuracy, detection F1,
> text F1). They illustrate each assignment's outcome and must **not** be
> compared against each other as a single ranking.

---

## ML / AI Workflow Coverage

The ten assignments collectively trace the course's practical arc:

| Concept | Assignments |
|---|---|
| Data preprocessing & cleaning | A3 |
| Regression | A1, A4, A9, A10 |
| Classification (incl. imbalanced) | A5, A8 |
| Computer vision (CNN, detection) | A6, A7 |
| NLP / sentiment | A8 |
| Feature selection & importance | A9 |
| Evolutionary / search methods | A2 |
| MLOps, tracking & deployment | A10 |

Course topics such as K-Means, PCA, SVM classification, XGBoost, transfer
learning and generative AI are covered in the TechPulse curriculum but are not
implemented by these ten lab assignments, so they are intentionally not
claimed here.

---

## Engineering Practices

Applied consistently across the assignments:

- **Reproducibility**: fixed random seeds (typically 42) for splits, models and generators
- **Leakage-aware preprocessing**: sklearn `Pipeline`/`ColumnTransformer` fitted on training data only; tokenizer fitted on training text only
- **Train/test separation**: stratified splits where class structure requires it
- **Automated testing**: 361 pytest tests in total across the ten suites
- **Evaluation discipline**: MAE/RMSE/R², precision/recall/F1, ROC-AUC/PR-AUC chosen per task; confusion matrices and error analysis
- **Artifact generation**: metrics JSON/CSV, plots, trained-model artifacts per assignment
- **Input validation**: Pydantic schemas on the A10 API boundary
- **Documentation**: per-assignment READMEs with measured results and stated limitations
- **MLOps hygiene**: health checks, model metadata, versioned API, `.gitignore`-excluded generated artifacts

---

## MLOps Workflow

Assignment 10 implements the lifecycle end-to-end:

```
Synthetic vehicle data (seed 42)
        ↓
Training (RandomForestRegressor)
        ↓
MLflow Tracking (params · metrics · model · artifacts)
        ↓
Model Artifact (joblib + metadata)
        ↓
FastAPI Service (/health · /version · /predict)
        ↓
Docker Image (model trained inside the image)
        ↓
Health Check (Docker HEALTHCHECK + scripts/healthcheck.py)
        ↓
Local Deployment Simulation (build → run → verify)
```

CI is provided by a GitHub Actions workflow
(`Assignment-10-MLOps-Workflow/.github/workflows/ci.yml`) that installs
dependencies, runs the test suite, validates the application import and
smoke-trains the model. Note: GitHub Actions only picks up workflows from the
repository-root `.github/workflows/` directory, so this file needs to be
located there (or referenced from it) for execution on GitHub. Details:
[Assignment 10](./Assignment-10-MLOps-Workflow/).

---

## Datasets

| Dataset | Used in | Notes |
|---|---|---|
| UCI Auto MPG | A1 | classic 398-row regression dataset |
| CarDekho "Car details v3" | A3, A4, A9 | 8,128 × 13 automotive dataset; reused read-only across assignments |
| AI4I 2020 Predictive Maintenance | A5 | 10,000 × 14 sensor logs, 3.39% failure rate |
| GTSRB | A6 | 39,209 train / 12,630 test images, 43 classes |
| INRIA Person | A7 | evaluation split with ground-truth pedestrian boxes |
| Amazon Automotive Reviews | A8 | 20,473 raw vehicle-customer reviews |
| Synthetic automotive dataset | A10 | 4,000 deterministic rows generated for the MLOps demo |

Each assignment downloads or locates its own data according to the
instructions in its own README; no dataset is bundled wholesale in this
repository.

---

## Repository Structure

```
Tata-TechPulse-Applied-AIML/
├── Assignment-01-Car-Mileage/
├── Assignment-02-Simulated-Driving-Agent/
├── Assignment-03-Data-Cleaning/
├── Assignment-04-Vehicle-Price-Prediction/
├── Assignment-05-Predictive-Maintenance/
├── Assignment-06-Traffic-Sign-Classification/
├── Assignment-07-Pedestrian-Detection/
├── Assignment-08-Sentiment-LSTM/
├── Assignment-09-Feature-Importance/
├── Assignment-10-MLOps-Workflow/
└── README.md
```

---

## Validation

Every assignment ships with an automated pytest suite; all ten suites pass.

| Assignment | Tests |
|---|---|
| A1 | 28/28 |
| A2 | 53/53 |
| A3 | 48/48 |
| A4 | 35/35 |
| A5 | 36/36 |
| A6 | 33/33 |
| A7 | 45/45 |
| A8 | 30/30 |
| A9 | 21/21 |
| A10 | 32/32 |
| **Total** | **361/361** |

---

## Reproducibility

Each assignment is self-contained: it has its own README, dependencies
(`requirements.txt` where applicable) and run commands, and can be executed
independently of the others. Experiments use fixed random seeds wherever the
task allows deterministic execution, and the per-assignment READMEs state
exactly what is reproducible (A10, for example, documents that metrics are
bit-identical across reruns while binary model files are not claimed to be).

Because the assignments use different (sometimes conflicting) library
requirements, there is deliberately **no single global requirements file** —
install dependencies per assignment.

General setup:

```bash
git clone https://github.com/Aarya2304/Tata-TechPulse-Applied-AIML.git
cd Tata-TechPulse-Applied-AIML
```

Example — run an assignment's tests:

```bash
cd Assignment-01-Car-Mileage
python -m pip install -r requirements.txt   # per-assignment deps
python -m pytest tests -q
```

Example — Assignment 10 (MLOps):

```bash
cd Assignment-10-MLOps-Workflow
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m src.train          # trains + logs to MLflow
docker compose up --build    # serves the API locally
```

---

## Notes

- All reported results were produced by actually running each assignment's
  pipeline; per-assignment READMEs contain the full measured tables.
- Metrics are dataset- and task-specific; cross-assignment comparison is not
  meaningful (see [Selected Results](#selected-results)).
- Feature-importance results (A9) indicate model dependence, not causality.
- Assignment 10 uses explicitly synthetic automotive data so the MLOps
  workflow can be demonstrated deterministically and quickly.
- The CI workflow is included and its steps were validated locally; GitHub
  Actions execution depends on the workflow being located in the
  repository-root `.github/workflows/` directory and on repository pushes.

---

## Author

**Aarya Yadav**
B.Tech — Computer Engineering, AI & Data Science
MIT World Peace University, Pune

Built as part of the Tata Technologies TechPulse FY-26 Applied AI/ML program.
