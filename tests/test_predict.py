"""Tests for src/models/predict.py, using a small synthetic fitted model."""

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_METHOD,
    CALIBRATOR_ARTIFACT_FILE,
    DECISION_THRESHOLD,
    MODEL_ARTIFACT_FILE,
)
from src.models.calibrate import (
    Calibrator,
    analytic_threshold,
    calibrator_artifact_path,
)
from src.models.predict import FEATURE_COLUMNS, FraudPredictor

# A known Platt correction: shrink the log-odds and shift them down, as a
# calibrator for an inflated model would.
PLATT_A, PLATT_B = 0.8, -2.0


@pytest.fixture
def artifact_paths(tmp_path):
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    # Make the target cleanly separable on Amount so predictions are deterministic.
    y = (X["Amount"] > 0).astype(int)

    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    calibrator_path = tmp_path / "calibrator.pkl"
    joblib.dump(model, model_path)
    joblib.dump(Calibrator(method="platt", platt_a=PLATT_A, platt_b=PLATT_B), calibrator_path)
    return model_path, calibrator_path


def _features(amount: float = 1.0) -> pd.DataFrame:
    features = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
    features["Amount"] = amount
    return features


@pytest.mark.parametrize("missing", ["model", "calibrator"])
def test_load_raises_clear_error_when_an_artifact_is_missing(artifact_paths, missing, tmp_path) -> None:
    model_path, calibrator_path = artifact_paths
    if missing == "model":
        model_path = tmp_path / "missing.pkl"
    else:
        calibrator_path = tmp_path / "missing.pkl"
    predictor = FraudPredictor(model_path=model_path, calibrator_path=calibrator_path)

    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_predict_lazy_loads_and_returns_expected_shape(artifact_paths) -> None:
    model_path, calibrator_path = artifact_paths
    predictor = FraudPredictor(model_path=model_path, calibrator_path=calibrator_path, threshold=0.5)

    result = predictor.predict(_features())

    assert predictor.model is not None and predictor.calibrator is not None  # lazy-loaded
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["threshold_applied"] == 0.5
    assert result["flagged"] == (result["fraud_probability"] >= 0.5)


def test_predict_returns_the_calibrated_probability(artifact_paths) -> None:
    model_path, calibrator_path = artifact_paths
    predictor = FraudPredictor(model_path=model_path, calibrator_path=calibrator_path)
    features = _features()

    result = predictor.predict(features)

    margin = joblib.load(model_path).predict(features[FEATURE_COLUMNS], output_margin=True)[0]
    expected = 1 / (1 + np.exp(-(PLATT_A * margin + PLATT_B)))
    assert result["fraud_probability"] == pytest.approx(expected)


def test_predict_reorders_columns_regardless_of_input_order(artifact_paths) -> None:
    model_path, calibrator_path = artifact_paths
    predictor = FraudPredictor(model_path=model_path, calibrator_path=calibrator_path, threshold=0.5)
    ordered = _features()
    shuffled = ordered[list(reversed(FEATURE_COLUMNS))]

    assert predictor.predict(ordered) == predictor.predict(shuffled)


def test_served_threshold_is_the_cost_derived_one() -> None:
    # The served score is a calibrated probability, so the threshold must be
    # C_FP / (C_FP + C_FN), not a number searched on some data slice.
    assert DECISION_THRESHOLD == pytest.approx(analytic_threshold())
    assert FraudPredictor().threshold == DECISION_THRESHOLD


def test_served_calibrator_matches_the_served_model() -> None:
    # The calibrator train.py saves for the served model is the one the API loads.
    assert MODEL_ARTIFACT_FILE.name == "model_unweighted.pkl"
    assert CALIBRATOR_ARTIFACT_FILE == calibrator_artifact_path("unweighted", CALIBRATION_METHOD)
