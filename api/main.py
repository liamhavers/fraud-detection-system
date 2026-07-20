"""FastAPI app exposing /predict and /health endpoints."""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from api.schemas import HealthResponse, PredictionResponse, TransactionRequest
from src.models.predict import FraudPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("fraud_api")

predictor = FraudPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup rather than on the first request, so a
    # missing artifact fails fast instead of surfacing on someone's first call.
    predictor.load()
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    logger.info(
        "Prediction request received: amount=%s time=%s", transaction.amount, transaction.time
    )

    features = pd.DataFrame(
        [
            {
                "Time": transaction.time,
                **{f"V{i}": getattr(transaction, f"v{i}") for i in range(1, 29)},
                "Amount": transaction.amount,
            }
        ]
    )
    result = predictor.predict(features)

    logger.info(
        "Prediction result: probability=%.4f flagged=%s threshold=%.2f",
        result["fraud_probability"],
        result["flagged"],
        result["threshold_applied"],
    )
    return PredictionResponse(**result)
