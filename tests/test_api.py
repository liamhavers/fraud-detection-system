"""Tests for the FastAPI app."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from api.main import app, predictor
from src.models.predict import FEATURE_COLUMNS

client = TestClient(app)

SAMPLE_TRANSACTION = {"time": 68.0, "amount": 2.69, **{f"v{i}": 0.0 for i in range(1, 29)}}


@pytest.fixture(autouse=True)
def stub_predictor_model():
    """Predict tests stub the model directly rather than relying on a real
    trained artifact — CI has no downloaded data/trained model to load, and
    `TestClient(app)` without a `with` block never triggers the app's
    lifespan (so `predictor.load()` is never called here), which is exactly
    what lets this stub take effect instead.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["Amount"] > 0).astype(int)
    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    predictor.model = model
    yield
    predictor.model = None


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_malformed_request() -> None:
    response = client.post("/predict", json={"amount": 100.0})
    assert response.status_code == 422


def test_predict_returns_expected_schema() -> None:
    response = client.post("/predict", json=SAMPLE_TRANSACTION)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"fraud_probability", "flagged", "threshold_applied"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert body["threshold_applied"] == predictor.threshold
