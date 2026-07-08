"""Pydantic request/response models for the fraud detection API."""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """A single transaction's features, matching the Kaggle dataset's PCA columns."""

    time: float = Field(..., description="Seconds elapsed since the first transaction")
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float = Field(..., ge=0, description="Transaction amount")


class PredictionResponse(BaseModel):
    fraud_probability: float
    flagged: bool
    threshold_applied: float


class HealthResponse(BaseModel):
    status: str
