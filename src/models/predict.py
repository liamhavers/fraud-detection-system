"""Inference wrapper used by the API: load model artifact, preprocess, predict."""

import joblib
import pandas as pd

from src.config import DECISION_THRESHOLD, MODEL_ARTIFACT_FILE

# Column names and order the model was trained on (Time, V1..V28, Amount) —
# the served request schema uses lowercase field names, so inputs are
# remapped into this exact shape before scoring.
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FraudPredictor:
    """Loads a trained model artifact once and serves predictions."""

    def __init__(self, model_path=MODEL_ARTIFACT_FILE, threshold: float = DECISION_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found. Run `python -m src.models.train` "
                "to produce a trained model artifact first."
            )
        self.model = joblib.load(self.model_path)

    def predict(self, features: pd.DataFrame) -> dict:
        """Return fraud probability, flagged boolean, and threshold applied."""
        if self.model is None:
            self.load()

        X = features[FEATURE_COLUMNS]
        fraud_probability = float(self.model.predict_proba(X)[:, 1][0])

        return {
            "fraud_probability": fraud_probability,
            "flagged": fraud_probability >= self.threshold,
            "threshold_applied": self.threshold,
        }
