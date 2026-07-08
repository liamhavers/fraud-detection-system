"""FastAPI app exposing /predict and /health endpoints."""

import logging

from fastapi import FastAPI

from api.schemas import HealthResponse, PredictionResponse, TransactionRequest
from src.models.predict import FraudPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

app = FastAPI(title="Fraud Detection API")
predictor = FraudPredictor()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    logger.info("Prediction request received: amount=%s", transaction.amount)
    raise NotImplementedError
