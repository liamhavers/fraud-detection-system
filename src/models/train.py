"""Training pipeline: imbalance handling (SMOTE vs class-weighting) and model fit."""

import pandas as pd


def train_with_class_weighting(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit an XGBoost model using scale_pos_weight for imbalance handling."""
    raise NotImplementedError


def train_with_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit an XGBoost model on SMOTE-resampled training data."""
    raise NotImplementedError
