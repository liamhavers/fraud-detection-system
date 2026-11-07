"""Inference wrapper used by the API: load model + calibrator, preprocess, predict."""

from pathlib import Path

import joblib
import pandas as pd

from src.config import CALIBRATOR_ARTIFACT_FILE, DECISION_THRESHOLD, MODEL_ARTIFACT_FILE
from src.models.calibrate import Calibrator

# Column names and order the model was trained on (Time, V1..V28, Amount) —
# the served request schema uses lowercase field names, so inputs are
# remapped into this exact shape before scoring.
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FraudPredictor:
    """Loads a trained model and its calibrator once and serves predictions.

    The returned `fraud_probability` is the *calibrated* probability (Platt
    scaling fitted on the validation slice), so it can be read as a real
    chance of fraud and compared directly against the cost-derived threshold.
    The model and calibrator must come from the same training run: the
    calibrator corrects that specific model's scores.
    """

    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_FILE,
        calibrator_path: Path = CALIBRATOR_ARTIFACT_FILE,
        threshold: float = DECISION_THRESHOLD,
    ):
        self.model_path = model_path
        self.calibrator_path = calibrator_path
        self.threshold = threshold
        self.model = None
        self.calibrator: Calibrator | None = None

    def load(self) -> None:
        for path in (self.model_path, self.calibrator_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `python -m src.models.train`, which saves "
                    "the model and its calibrator together."
                )
        self.model = joblib.load(self.model_path)
        self.calibrator = joblib.load(self.calibrator_path)

    def predict(self, features: pd.DataFrame) -> dict:
        """Return calibrated fraud probability, flagged boolean, and threshold applied."""
        if self.model is None or self.calibrator is None:
            self.load()

        X = features[FEATURE_COLUMNS]
        fraud_probability = float(self.calibrator.predict_proba(self.model, X)[0])

        return {
            "fraud_probability": fraud_probability,
            "flagged": fraud_probability >= self.threshold,
            "threshold_applied": self.threshold,
        }
