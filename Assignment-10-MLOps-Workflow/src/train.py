"""Training + MLflow experiment tracking.

``python -m src.train`` executes the complete training workflow:

1. generate the deterministic synthetic dataset
2. train the RandomForestRegressor
3. evaluate on a held-out test split
4. log params / metrics / model / artifacts to local MLflow (mlruns/)
5. export the deployment artifact (joblib) + model_metadata.json

The exported joblib artifact is the *deployment* copy; the MLflow run is
the *provenance* record.  Both are produced by this single command.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src import config, model as model_module
from src.data import dataset_summary, generate_dataset, load_training_data


def _utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 (metadata requirement)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def train_model(n_samples: int, n_estimators: int, seed: int):
    """Train the regressor and return the fitted model with its metrics.

    Deterministic: identical inputs always produce identical metrics.
    """
    X, y = load_training_data(n_samples=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=seed
    )

    reg = model_module.build_model(n_estimators=n_estimators, random_state=seed)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    metrics = model_module.evaluate_regression(y_test, y_pred)
    return reg, metrics, (X_train, X_test, y_train, y_test)


def train_and_log(
    n_samples: int = config.N_SAMPLES,
    n_estimators: int = config.N_ESTIMATORS,
    seed: int = config.RANDOM_STATE,
    tracking_uri: str | None = None,
) -> dict:
    """Run the full training workflow with MLflow tracking.

    Returns a summary dict containing the metrics, artifact paths and the
    MLflow run ID.  All parameters, metrics, the trained model and small
    JSON/text artifacts are logged to the local file-based MLflow store.
    """
    mlflow.set_tracking_uri(tracking_uri or config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Ensure the artifact directory exists before any writes (it is not
    # present in a fresh clone or the Docker build context).
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    training_started = time.time()
    with mlflow.start_run(run_name=config.RUN_NAME) as run:
        reg, metrics, (X_train, X_test, y_train, y_test) = train_model(
            n_samples=n_samples, n_estimators=n_estimators, seed=seed
        )

        # ---- parameters -------------------------------------------------
        mlflow.log_params(
            {
                "model_type": config.MODEL_TYPE,
                "n_estimators": n_estimators,
                "random_state": seed,
                "test_size": config.TEST_SIZE,
                "n_features": len(config.FEATURE_NAMES),
                "dataset_size": int(len(X_train) + len(X_test)),
                "target": config.TARGET_NAME,
                "app_version": config.APP_VERSION,
            }
        )
        # ---- metrics ----------------------------------------------------
        mlflow.log_metrics(metrics)

        # ---- model artifact (MLflow sklearn flavour) --------------------
        # skops_trusted_types: mlflow >= 3 validates the persisted model
        # with skops and refuses RandomForest's internal Tree node-storage
        # type unless explicitly trusted.  The model is trained locally in
        # this project, so trusting exactly this one reviewed type is safe.
        mlflow.sklearn.log_model(
            sk_model=reg,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # ---- small metadata artifacts ------------------------------------
        summary = dataset_summary(generate_dataset(n_samples=n_samples, seed=seed))
        feature_info = {
            "feature_names": list(config.FEATURE_NAMES),
            "target_name": config.TARGET_NAME,
            "n_features": len(config.FEATURE_NAMES),
            "dataset_summary": summary,
        }
        feature_info_path = config.ARTIFACTS_DIR / "feature_info.json"
        feature_info_path.write_text(
            json.dumps(feature_info, indent=2), encoding="utf-8"
        )

        metrics_payload = {
            "metrics": metrics,
            "n_test_samples": int(len(X_test)),
            "training_seconds": round(time.time() - training_started, 3),
            "generated_at_utc": _utc_now_iso(),
        }
        metrics_path = config.METRICS_PATH
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )

        evaluation_text = (
            "Vehicle price regression - holdout evaluation\n"
            f"model_type={config.MODEL_TYPE} n_estimators={n_estimators} "
            f"random_state={seed}\n"
            f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
            f"R2={metrics['r2']:.4f}\n"
        )
        eval_path = config.EVALUATION_PATH
        eval_path.write_text(evaluation_text, encoding="utf-8")

        mlflow.log_artifacts(str(config.ARTIFACTS_DIR), artifact_path="reports")

        run_id = run.info.run_id

    # ---- deployment export (outside the MLflow run) ----------------------
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 keeps the 100-tree forest small enough for the repo/image.
    joblib.dump(reg, config.MODEL_PATH, compress=3)
    metadata = {
        "model_type": config.MODEL_TYPE,
        "random_state": seed,
        "n_estimators": n_estimators,
        "training_timestamp": _utc_now_iso(),
        "dataset_type": config.DATASET_TYPE,
        "dataset_size": int(len(X_train) + len(X_test)),
        "n_features": len(config.FEATURE_NAMES),
        "feature_names": list(config.FEATURE_NAMES),
        "target_name": config.TARGET_NAME,
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_experiment": config.EXPERIMENT_NAME,
        "application_version": config.APP_VERSION,
    }
    config.METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model_path": str(config.MODEL_PATH.relative_to(config.PROJECT_ROOT)),
        "metadata_path": str(config.METADATA_PATH.relative_to(config.PROJECT_ROOT)),
    }


def main() -> dict:
    """Entry point for ``python -m src.train``.

    ``--smoke`` runs the same workflow with a small dataset/model so CI
    and the Docker build finish quickly; the default run uses the full
    configured sizes.
    """
    smoke = "--smoke" in sys.argv
    n_samples = config.SMOKE_N_SAMPLES if smoke else config.N_SAMPLES
    n_estimators = config.SMOKE_N_ESTIMATORS if smoke else config.N_ESTIMATORS
    print(
        "=== TechPulse Assignment 10 - training + MLflow tracking ===\n"
        f"mode={('smoke' if smoke else 'full')} "
        f"n_samples={n_samples} n_estimators={n_estimators}"
    )
    result = train_and_log(
        n_samples=n_samples, n_estimators=n_estimators, seed=config.RANDOM_STATE
    )
    m = result["metrics"]
    print(
        f"run_id        : {result['run_id']}\n"
        f"MAE / RMSE / R2: {m['mae']:.2f} / {m['rmse']:.2f} / {m['r2']:.4f}\n"
        f"model artifact : {result['model_path']}\n"
        f"metadata       : {result['metadata_path']}\n"
        "MLflow UI: mlflow ui --backend-store-uri ./mlruns"
    )
    return result


if __name__ == "__main__":
    main()
