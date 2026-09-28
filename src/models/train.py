"""Training pipeline: imbalance handling (SMOTE vs class-weighting) and model fit."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import polars as pl
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    EXPERIMENTS_DIR,
    IEEE_MODEL_ARTIFACT_FILE,
    MODEL_ARTIFACT_CLASS_WEIGHTED_FILE,
    MODEL_ARTIFACT_SMOTE_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    XGB_PARAMS,
)
from src.data.load import scan_raw_data, scan_raw_ieee_data
from src.data.preprocess import clean, time_aware_split
from src.data.preprocess_ieee import (
    engineer_features,
    join_transaction_identity,
)
from src.data.preprocess_ieee import (
    time_aware_split as ieee_time_aware_split,
)
from src.models.evaluate import pr_auc_score


def train_baseline_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit a class-weighted logistic regression as a simple reference point.

    Features are standardised first: Kaggle's `Time`/`Amount` columns are on a
    much larger scale than the PCA components, which otherwise stalls LBFGS
    convergence (XGBoost is scale-invariant so this isn't needed elsewhere).
    """
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_with_class_weighting(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit an XGBoost model using scale_pos_weight for imbalance handling."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model


def train_with_smote(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit an XGBoost model on SMOTE-resampled training data."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_resampled, y_resampled)
    return model


def log_experiment(name: str, params: dict, metrics: dict) -> Path:
    """Append a run record to experiments/ as a timestamped JSON file."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = EXPERIMENTS_DIR / f"{timestamp}_{name}.json"
    record = {
        "name": name,
        "timestamp": timestamp,
        "params": params,
        "metrics": metrics,
    }
    run_path.write_text(json.dumps(record, indent=2))
    return run_path


def _run_kaggle_training() -> dict:
    # Preprocessing runs in polars; the frames cross over to pandas here,
    # at the boundary with scikit-learn / imbalanced-learn.
    train_df, test_df = (
        df.to_pandas() for df in pl.collect_all(time_aware_split(clean(scan_raw_data())))
    )
    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    baseline = train_baseline_logistic_regression(X_train, y_train)
    baseline_pr_auc = pr_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

    class_weighted_model = train_with_class_weighting(X_train, y_train)
    class_weighted_pr_auc = pr_auc_score(
        y_test, class_weighted_model.predict_proba(X_test)[:, 1]
    )

    smote_model = train_with_smote(X_train, y_train)
    smote_pr_auc = pr_auc_score(y_test, smote_model.predict_proba(X_test)[:, 1])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(class_weighted_model, MODEL_ARTIFACT_CLASS_WEIGHTED_FILE)
    joblib.dump(smote_model, MODEL_ARTIFACT_SMOTE_FILE)

    metrics = {
        "baseline_logistic_regression_pr_auc": baseline_pr_auc,
        "xgboost_class_weighted_pr_auc": class_weighted_pr_auc,
        "xgboost_smote_pr_auc": smote_pr_auc,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    log_experiment("kaggle_creditcard", params=XGB_PARAMS, metrics=metrics)
    return metrics


def _run_ieee_training() -> dict:
    # Materialise the joined, sorted split once: fitting the encoders needs
    # a collect, and without this both it and the final collect would
    # re-scan and re-join the raw CSVs.
    train_df, test_df = pl.collect_all(
        ieee_time_aware_split(join_transaction_identity(*scan_raw_ieee_data()))
    )
    train_fe, test_fe = (
        df.to_pandas()
        for df in pl.collect_all(engineer_features(train_df.lazy(), test_df.lazy()))
    )

    drop_cols = ["TransactionID", "isFraud"]
    X_train, y_train = train_fe.drop(columns=drop_cols), train_fe["isFraud"]
    X_test, y_test = test_fe.drop(columns=drop_cols), test_fe["isFraud"]

    model = train_with_class_weighting(X_train, y_train)
    test_pr_auc = pr_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, IEEE_MODEL_ARTIFACT_FILE)

    metrics = {
        "xgboost_class_weighted_pr_auc": test_pr_auc,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    log_experiment("ieee_cis", params=XGB_PARAMS, metrics=metrics)
    return metrics


if __name__ == "__main__":
    print("Training Kaggle Credit Card Fraud models (baseline LR, XGBoost class-weighted, XGBoost SMOTE)...")
    kaggle_metrics = _run_kaggle_training()
    for key, value in kaggle_metrics.items():
        print(f"  {key}: {value}")

    print("\nTraining IEEE-CIS XGBoost model on engineered features...")
    ieee_metrics = _run_ieee_training()
    for key, value in ieee_metrics.items():
        print(f"  {key}: {value}")
