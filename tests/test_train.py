"""Tests for src/models/train.py, using small synthetic imbalanced data."""

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.train import (
    log_experiment,
    train_baseline_logistic_regression,
    train_with_class_weighting,
    train_with_smote,
)


def _make_synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n_majority, n_minority = 200, 20

    majority = rng.normal(loc=0.0, scale=1.0, size=(n_majority, 2))
    minority = rng.normal(loc=4.0, scale=1.0, size=(n_minority, 2))

    X = pd.DataFrame(np.vstack([majority, minority]), columns=["f1", "f2"])
    y = pd.Series([0] * n_majority + [1] * n_minority)
    return X, y


def test_train_with_class_weighting_fits_and_separates_classes() -> None:
    X, y = _make_synthetic_data()
    model = train_with_class_weighting(X, y)

    assert isinstance(model, XGBClassifier)
    proba = model.predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_train_with_smote_fits_and_separates_classes() -> None:
    X, y = _make_synthetic_data()
    model = train_with_smote(X, y)

    assert isinstance(model, XGBClassifier)
    proba = model.predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_train_baseline_logistic_regression_returns_fitted_pipeline() -> None:
    X, y = _make_synthetic_data()
    model = train_baseline_logistic_regression(X, y)

    assert isinstance(model, Pipeline)
    proba = model.predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_log_experiment_writes_expected_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.models.train.EXPERIMENTS_DIR", tmp_path)

    run_path = log_experiment(
        "unit_test_run", params={"n_estimators": 10}, metrics={"pr_auc": 0.5}
    )

    assert run_path.exists()
    record = json.loads(run_path.read_text())
    assert record["name"] == "unit_test_run"
    assert record["params"] == {"n_estimators": 10}
    assert record["metrics"] == {"pr_auc": 0.5}
