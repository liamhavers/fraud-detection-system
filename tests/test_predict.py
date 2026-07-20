"""Tests for src/models/predict.py, using a small synthetic fitted model."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.models.predict import FEATURE_COLUMNS, FraudPredictor


@pytest.fixture
def fitted_model_path(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    # Make the target cleanly separable on Amount so predictions are deterministic.
    y = (X["Amount"] > 0).astype(int)

    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    import joblib

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    return model_path


def test_load_raises_clear_error_when_artifact_missing(tmp_path) -> None:
    predictor = FraudPredictor(model_path=tmp_path / "missing.pkl")

    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_predict_lazy_loads_and_returns_expected_shape(fitted_model_path) -> None:
    predictor = FraudPredictor(model_path=fitted_model_path, threshold=0.5)
    features = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
    features["Amount"] = 1.0

    result = predictor.predict(features)

    assert predictor.model is not None  # lazy-loaded
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["threshold_applied"] == 0.5
    assert result["flagged"] == (result["fraud_probability"] >= 0.5)


def test_predict_reorders_columns_regardless_of_input_order(fitted_model_path) -> None:
    predictor = FraudPredictor(model_path=fitted_model_path, threshold=0.5)
    ordered = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
    ordered["Amount"] = 1.0
    shuffled = ordered[list(reversed(FEATURE_COLUMNS))]

    assert predictor.predict(ordered) == predictor.predict(shuffled)
