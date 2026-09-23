"""FastAPI prediction service for the trained vehicle-price model.

Endpoints:
    GET /health   -> {"status": "healthy", ...}
    GET /version  -> {"version": "1.0.0"}
    POST /predict -> validated feature payload -> price prediction

Input is validated with Pydantic; invalid/missing fields produce a
useful 422 response.  No internal filesystem paths are exposed.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config, inference

app = FastAPI(
    title="TechPulse Vehicle Price API",
    description="Assignment 10 - MLOps workflow simulation (prediction service)",
    version=config.APP_VERSION,
)


class VehicleFeatures(BaseModel):
    """Request schema with per-field validation (Pydantic v2)."""

    vehicle_age: int = Field(ge=0, le=80, description="vehicle age in years")
    km_driven: int = Field(ge=0, le=2_000_000, description="total km driven")
    mileage_kmpl: float = Field(gt=0, le=100, description="mileage in km/l")
    engine_cc: int = Field(gt=0, le=10_000, description="engine displacement cc")
    max_power_bhp: float = Field(gt=0, le=2_000, description="max power in bhp")


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    model_type: str


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe (used by Docker + deploy script)."""
    return {
        "status": "healthy",
        "version": config.APP_VERSION,
        "model_type": config.MODEL_TYPE,
    }


@app.get("/version")
def version() -> dict:
    """Expose the central application version."""
    return {"version": config.APP_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(vehicle: VehicleFeatures) -> PredictionResponse:
    """Predict the selling price for one vehicle."""
    try:
        price = inference.predict_price(vehicle.model_dump())
    except Exception as exc:  # narrow guard: model errors -> HTTP error
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    return PredictionResponse(
        prediction=round(price, 2),
        model_version=config.APP_VERSION,
        model_type=config.MODEL_TYPE,
    )
