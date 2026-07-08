"""Tests for the FastAPI app."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_malformed_request() -> None:
    response = client.post("/predict", json={"amount": 100.0})
    assert response.status_code == 422
