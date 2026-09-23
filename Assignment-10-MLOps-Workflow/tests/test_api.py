"""Tests for the FastAPI application (in-process via TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import config, inference
from src.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Ensure the model artifact exists before the app serves requests."""
    inference.reset_model_cache()
    inference.ensure_model()
    return TestClient(app)


VALID_PAYLOAD = {
    "vehicle_age": 5,
    "km_driven": 45_000,
    "mileage_kmpl": 18.5,
    "engine_cc": 1_498,
    "max_power_bhp": 100,
}


def test_health_returns_200_and_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == config.APP_VERSION


def test_version_endpoint_returns_central_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": config.APP_VERSION}


def test_predict_accepts_valid_input(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], (int, float))
    assert body["prediction"] > 0
    assert body["model_version"] == config.APP_VERSION
    assert body["model_type"] == config.MODEL_TYPE


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "max_power_bhp"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_rejects_invalid_type(client):
    response = client.post("/predict", json={**VALID_PAYLOAD, "vehicle_age": "five"})
    assert response.status_code == 422


def test_predict_rejects_out_of_range_values(client):
    response = client.post("/predict", json={**VALID_PAYLOAD, "vehicle_age": 999})
    assert response.status_code == 422
    response = client.post("/predict", json={**VALID_PAYLOAD, "mileage_kmpl": -1})
    assert response.status_code == 422


def test_predict_accepts_boundary_values(client):
    payload = {**VALID_PAYLOAD, "vehicle_age": 0, "km_driven": 500}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_is_consistent_between_calls(client):
    first = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
    second = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
    assert first == second


def test_openapi_schema_lists_expected_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths
    assert "/version" in paths
    assert "/predict" in paths
