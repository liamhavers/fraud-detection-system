"""Inference wrapper used by the API: load model artifact, preprocess, predict."""

import pandas as pd

from src.config import DECISION_THRESHOLD, MODEL_ARTIFACT_FILE


class FraudPredictor:
    """Loads a trained model artifact once and serves predictions."""

    def __init__(self, model_path=MODEL_ARTIFACT_FILE, threshold: float = DECISION_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> dict:
        """Return fraud probability, flagged boolean, and threshold applied."""
        raise NotImplementedError
