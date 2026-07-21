"""
Serving layer for the car price model.

This is the online half of the system. It never trains and never reads the
CSV. At startup it loads the artifact written by train.py -- the fitted model
plus every preprocessing parameter learned during training -- and from then on
it only answers requests.

The API takes raw, human-readable car specs. The log transforms, encoding and
scaling all happen internally, and the prediction is converted back out of log
space before being returned, so callers never need to know the target is logged.
"""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocess import transform

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

# Loaded once at startup, not per request. Unpickling on every call would add
# hundreds of milliseconds of latency for no reason.
artifact = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No model artifact at {MODEL_PATH}. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        artifact["model"], artifact["params"] = pickle.load(f).values()
    yield
    artifact.clear()


app = FastAPI(
    title="Car Price Prediction API",
    description="Predicts a car's MSRP from its specifications.",
    version="1.0.0",
    lifespan=lifespan,
)


class CarRequest(BaseModel):
    """
    One car's specs. Only make, model and year are required -- anything else
    left out falls back to the imputation rules learned during training, the
    same way a row with missing values would have been handled there.
    """

    make: str = Field(..., examples=["BMW"])
    model: str = Field(..., examples=["X5"])
    year: int = Field(..., ge=1900, le=2100, examples=[2016])

    engine_fuel_type: Optional[str] = Field(None, examples=["premium unleaded (required)"])
    engine_hp: Optional[float] = Field(None, ge=0, examples=[300])
    engine_cylinders: Optional[float] = Field(None, ge=0, examples=[6])
    transmission_type: Optional[str] = Field(None, examples=["AUTOMATIC"])
    driven_wheels: Optional[str] = Field(None, examples=["all wheel drive"])
    number_of_doors: Optional[float] = Field(None, ge=0, examples=[4])
    market_category: Optional[str] = Field(None, examples=["Luxury,Performance"])
    vehicle_size: Optional[str] = Field(None, examples=["Midsize"])
    vehicle_style: Optional[str] = Field(None, examples=["4dr SUV"])
    highway_mpg: Optional[float] = Field(None, gt=0, examples=[27])
    city_mpg: Optional[float] = Field(None, gt=0, examples=[18])
    popularity: Optional[float] = Field(None, gt=0, examples=[3916])


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"


@app.get("/health")
def health():
    """Liveness probe. Container orchestrators hit this to decide if the
    service is up, so it must not depend on anything that can hang."""
    return {"status": "ok", "model_loaded": bool(artifact)}


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarRequest):
    try:
        X = transform(car.model_dump(), artifact["params"])

        # The model was trained on log(msrp), so its output is in log space.
        # exp() converts it back to dollars before it leaves the service.
        log_price = artifact["model"].predict(X)[0]
        price = float(np.exp(log_price))

    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process car: {exc}")

    return PredictionResponse(predicted_price=round(price, 2))