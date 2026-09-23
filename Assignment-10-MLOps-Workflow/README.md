# Assignment 10 — MLOps Workflow Simulation

## 1. Objective

Build a small, fully reproducible MLOps project that demonstrates the complete
machine-learning lifecycle — training, experiment tracking, packaging, serving,
containerization, continuous integration and local deployment validation — for
the TechPulse FY-26 Applied AI/ML program.

## 2. Official Requirement

Official lab statement: **"Build a CI/CD pipeline using MLflow and Docker for
model deployment."**

This project implements that statement as a local, educational workflow: MLflow
tracks the experiment, Docker packages the prediction service, and GitHub
Actions provides CI. No cloud services, Kubernetes, or paid infrastructure are
involved.

## 3. MLOps Workflow Overview

```
Synthetic vehicle data (deterministic, seed=42)
  ↓
Training (sklearn RandomForestRegressor)
  ↓
MLflow Tracking (params + metrics + model + artifacts → ./mlruns)
  ↓
Model Artifact (joblib, compressed) + model_metadata.json
  ↓
FastAPI Inference Service (/health, /version, /predict)
  ↓
Docker Image (model trained inside the image at build time)
  ↓
Local Deployment (scripts/deploy.py: build → run → health → predict)
  ↓
Health Check (scripts/healthcheck.py)
```

CI side:

```
GitHub Push / PR
       ↓
Install Dependencies (requirements.txt + requirements-dev.txt)
       ↓
Run Tests (pytest)
       ↓
Validate Application (import check)
       ↓
Smoke Training + Artifact Verification
```

**CI vs CD in this project**

- **CI (continuous integration)** = automated validation: every push/PR runs
  the test suite, verifies the application imports, and performs a lightweight
  training validation (`.github/workflows/ci.yml`).
- **CD (continuous deployment)** = packaging and deployment *validation*: the
  Docker image is built, run locally, and its health and prediction endpoints
  are verified before success is reported (`scripts/deploy.py`).

This is a **local deployment simulation** — the project does not deploy to any
cloud provider, and no container registry credentials are used.

## 4. Dataset

- **Type**: deterministic **synthetic** vehicle dataset generated in code
  (`src/data.py`) with NumPy (`np.random.default_rng(seed=42)`).
- **Size**: 4,000 rows, 5 features + 1 target.
- **Features**: `vehicle_age`, `km_driven`, `mileage_kmpl`, `engine_cc`,
  `max_power_bhp`.
- **Target**: `selling_price` (INR).

Designed relationships match used-vehicle market intuition: higher max power
and engine size → higher price; newer vehicles and lower km_driven → higher
price; mileage contributes positively. Multiplicative log-normal noise keeps
prices positive and right-skewed.

> **Important**: this dataset is **synthetic** and exists only to demonstrate
> the MLOps workflow. It is **not** a real market-price dataset, and the
> numeric results below describe the synthetic process, not the used-car
> market. A CI/`--smoke` variant (600 rows, 30 trees) is used for fast tests
> and inside the Docker image.

## 5. Model

`RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` from
scikit-learn, with an 80/20 `train_test_split(random_state=42)`. The feature
order is fixed in `src/config.py`, making training fully deterministic.

## 6. MLflow Experiment Tracking

Local, file-based tracking — no remote server required.

- Experiment: **`TechPulse-Assignment-10`**
- Tracking URI: `file://.../mlruns` (set in `src/config.py`)
- Logged **parameters**: `model_type`, `n_estimators`, `random_state`,
  `test_size`, `n_features`, `dataset_size`, `target`, `app_version`
- Logged **metrics**: `mae`, `rmse`, `r2`
- Logged **model**: the fitted RandomForest via `mlflow.sklearn.log_model`
  (sklearn flavour, with input example and schema)
- Logged **artifacts**: `feature_info.json`, `model_metrics.json`,
  `evaluation.txt` (under `reports/` inside the run)
- Every run's ID is recorded in `artifacts/model_metadata.json`

mlflow 3 note: the filesystem backend is in maintenance mode, so the project
sets `MLFLOW_ALLOW_FILE_STORE=true` (in `src/__init__.py`, the Dockerfile, and
the Makefile UI target) to explicitly opt in to the assignment's required
local `mlruns/` store.

## 7. Model Packaging

Two artifacts are produced by `python -m src.train`:

1. **MLflow run** (provenance): the model in MLflow's sklearn format, loadable
   with `mlflow.sklearn.load_model("runs:/<run_id>/model")`.
2. **Deployment artifact** (serving): `artifacts/vehicle_price_rf.joblib`
   (joblib, `compress=3`, ≈ **6.1 MB**), described by
   `artifacts/model_metadata.json` (model type, seed, timestamp in UTC
   ISO-8601, dataset type, feature names, metrics, MLflow run ID,
   application version).

`src/inference.py` loads the joblib artifact. If it is missing (fresh clone or
container build), it deterministically trains the small smoke model first, so
the API never depends on a model sitting only on the developer's machine.

## 8. FastAPI Inference Service

`src/app.py` exposes:

| Endpoint       | Method | Purpose                                        |
|----------------|--------|------------------------------------------------|
| `/health`      | GET    | Liveness/readiness: status, version, model type |
| `/version`     | GET    | Central application version (`1.0.0`)           |
| `/predict`     | POST   | Validated feature payload → price prediction    |

`/predict` input is validated with **Pydantic** (per-field bounds, e.g.
`0 ≤ vehicle_age ≤ 80`, `0 < mileage_kmpl ≤ 100`); invalid or missing fields
produce an HTTP 422 error with useful detail. No internal filesystem paths are
exposed in any response.

## 9. Docker Containerization

The `Dockerfile` uses `python:3.11-slim`, installs dependencies, copies only
`src/` + `scripts/` (tests and dev files are excluded via `.dockerignore`),
**trains the smoke model inside the image at build time** (`RUN python -m
src.train --smoke`) so the container is self-contained, exposes port 8000, and
starts uvicorn:

```
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

A Docker `HEALTHCHECK` runs `scripts/healthcheck.py` against the in-container
API. Built image: `techpulse-mlops:1.0.0`.

## 10. CI Pipeline

`.github/workflows/ci.yml` triggers on every **push** and **pull_request** and
runs on Python 3.11 (ubuntu-latest):

1. Check out the repository
2. Set up Python with pip caching
3. Install `requirements.txt` + `requirements-dev.txt`
4. Run `pytest -q`
5. Validate the application imports (`from src.app import app`)
6. Run smoke training (`python -m src.train --smoke`)
7. Verify `artifacts/model_metadata.json`, `artifacts/model_metrics.json`
   exist and an MLflow experiment directory was created

No DockerHub or cloud credentials are required.

## 11. CD / Local Deployment Simulation

`scripts/deploy.py` performs the deployment sequence against the local Docker
engine:

1. `docker build -t techpulse-mlops:1.0.0 .`
2. Remove any stale container, then `docker run -d --name techpulse-ml-api -p 8000:8000 ...`
3. Poll `GET /health` until the service responds `200` with `status=healthy`
4. Send one validation `POST /predict` and check the response is numeric
5. Report `DEPLOYMENT SUCCESS` (or clean up and exit non-zero on failure)

Run it with `python scripts/deploy.py` (cross-platform; works on Windows
without a shell script) or `make deploy`. `python scripts/deploy.py --verify`
only validates an already-running container. **This is explicitly a local
deployment simulation — no cloud deployment is performed.**

## 12. Project Structure

```
Assignment-10-MLOps-Workflow/
├── README.md
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # test dependencies
├── .gitignore                # excludes mlruns content, artifacts, caches, secrets
├── Dockerfile                # self-contained prediction-service image
├── docker-compose.yml        # single-service local deployment + healthcheck
├── .dockerignore             # keeps the build context minimal
├── Makefile                  # train / test / app / docker / deploy / ui targets
├── conftest.py, pytest.ini
├── src/
│   ├── __init__.py           # sets MLflow file-store opt-in before imports
│   ├── config.py             # version, seeds, params, MLflow, paths
│   ├── data.py               # deterministic synthetic dataset
│   ├── model.py              # model construction + MAE/RMSE/R²
│   ├── train.py              # training + MLflow logging + export
│   ├── inference.py          # model loading + prediction (+ bootstrap)
│   └── app.py                # FastAPI service
├── scripts/
│   ├── train_and_log.py      # standalone train+log entry point
│   ├── healthcheck.py        # /health validation (exit 0/1)
│   └── deploy.py             # local CD simulation (build/run/verify)
├── tests/                    # 32 tests (no Docker, no live server needed)
├── .github/workflows/ci.yml  # CI pipeline
├── mlruns/                   # local MLflow store (contents gitignored)
└── artifacts/                # model + metadata (contents gitignored)
```

## 13. Setup

```bash
cd Assignment-10-MLOps-Workflow
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Python 3.11 was used for the results below. The project is independently
runnable and contains no absolute paths.

## 14. Training

```bash
python -m src.train            # full run (4,000 samples, 100 trees)
python -m src.train --smoke    # fast run (600 samples, 30 trees)
```

Each run trains, evaluates, logs everything to MLflow, and (re)writes the
deployment artifact + metadata. Latest full run ID:
`178337f9ce5647179dc3d04ba0b8cb33`.

## 15. MLflow UI

```bash
# Linux/macOS (or Git Bash):
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns

# Windows PowerShell:
$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri ./mlruns

# or simply:  make ui
```

Then open <http://127.0.0.1:5000>. The UI reads the same `mlruns/` directory
the training script writes to (verified: the experiment and all runs are
listed). The `MLFLOW_ALLOW_FILE_STORE=true` flag is required because mlflow 3
places the filesystem backend in maintenance mode.

## 16. Running API Locally

```bash
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
# or:  make app
```

## 17. Running with Docker

```bash
docker build -t techpulse-mlops:1.0.0 .
docker run -d --name techpulse-ml-api -p 8000:8000 techpulse-mlops:1.0.0
docker ps --filter name=techpulse-ml-api   # → Up (healthy)
```

## 18. Running with Docker Compose

```bash
docker compose up --build      # then: http://localhost:8000/health
docker compose down
```

The compose file defines one service (`ml-api`) with a `/health`-based
healthcheck. If a container named `techpulse-ml-api` already exists from a
previous run, remove it first (`docker rm -f techpulse-ml-api`).

## 19. Running Tests

```bash
python -m pytest tests -q
```

32 tests, ≈ 30–45 s, no Docker and no live server required.

## 20. CI/CD Workflow

See sections 10 (CI) and 11 (CD). Summary:

- **CI** runs automatically on push/PR: tests → import validation → smoke
  training → artifact checks.
- **CD simulation** runs on demand: `python scripts/deploy.py` builds the
  image, starts the container, waits for health, and validates one prediction.

## 21. Example API Requests

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_age": 5,
    "km_driven": 45000,
    "mileage_kmpl": 18.5,
    "engine_cc": 1498,
    "max_power_bhp": 100
  }'
```

Response (full model):

```json
{"prediction": 517365.0, "model_version": "1.0.0", "model_type": "RandomForestRegressor"}
```

Health and version:

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"1.0.0","model_type":"RandomForestRegressor"}
curl http://localhost:8000/version
# {"version":"1.0.0"}
```

Invalid input is rejected:

```bash
curl -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"vehicle_age": 5}'
# 422
```

## 22. Results

Measured with the full configuration (4,000 samples, 100 trees, seed 42,
80/20 split; n_test = 800):

| Metric | Value |
|---|---:|
| MAE | 125,252.34 INR |
| RMSE | 243,943.00 INR |
| R² | 0.8167 |

Reproducibility: two consecutive `python -m src.train` runs produced
**identical metrics** (125,252.34 / 243,943.00 / 0.8167) with distinct MLflow
run IDs. Test suite: **32/32 passed**. Deployment simulation: exit code 0
(build → run → healthy → numeric prediction). All numbers above were produced
by actual runs of this project on the synthetic dataset described in section 4.

## 23. Limitations

- The dataset is **synthetic**; metrics characterize the generator, not the
  real used-vehicle market.
- The model is deliberately simple (no feature engineering, no tuning); it
  demonstrates the MLOps lifecycle, not state-of-the-art price prediction.
- File-based MLflow tracking is in maintenance mode in mlflow ≥ 3 (the project
  opts in explicitly); a SQLite/DB backend would be needed for team-scale
  tracking.
- The container image is ~1.28 GB because mlflow brings a large dependency
  tree; a production service could install only `fastapi`/`uvicorn`/
  `scikit-learn`/`joblib` and load the joblib artifact directly.
- CI runs tests but does not build the Docker image (kept fast and
  credential-free); image building is validated locally via the deploy script.
- The "deployment" is a local simulation; there is no registry push, staging
  environment, or rollback mechanism.

## 24. Reproducibility

- `RANDOM_STATE = 42` everywhere (dataset, split, forest).
- Identical metrics across repeated full training runs (verified above).
- The Docker build trains its model deterministically inside the image, so the
  serving stack is reproducible from source alone.
- Timestamps in metadata are UTC ISO-8601; the application version is pinned
  centrally (`1.0.0` in `src/config.py`).
- Model files are *not* claimed to be byte-identical across runs (joblib
  output can embed non-deterministic metadata); the verified reproducibility
  claim covers metrics and predictions, which were checked exactly.

## Security / Responsible MLOps Hygiene

- No secrets, API keys, or credentials anywhere in the repository.
- `.gitignore` excludes generated model artifacts, MLflow runs, caches,
  virtual environments, logs and `.env` files — the repository stays
  lightweight.
- Input validation on the API boundary (Pydantic); internal paths never leak
  into responses.
- Deterministic training; health checks on the container and deployment path;
  automated tests gate every commit via CI.
