"""Single source of truth for paths, thresholds, and model hyperparameters."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
REPORTS_DIR = ROOT_DIR / "reports"

# Kaggle Credit Card Fraud dataset — backs the served model.
CREDITCARD_RAW_DIR = DATA_RAW_DIR / "creditcard"
RAW_DATA_FILE = CREDITCARD_RAW_DIR / "creditcard.csv"
TRAIN_DATA_FILE = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_FILE = DATA_PROCESSED_DIR / "test.csv"
MODEL_ARTIFACT_FILE = MODELS_DIR / "model.pkl"
# Both imbalance-handling variants are kept around (not just the winner) so
# the Phase 3 shadow-mode A/B test has two real trained policies to compare.
MODEL_ARTIFACT_CLASS_WEIGHTED_FILE = MODELS_DIR / "model_class_weighted.pkl"
MODEL_ARTIFACT_SMOTE_FILE = MODELS_DIR / "model_smote.pkl"

# IEEE-CIS dataset — feature-engineering showcase, not served via the API.
IEEE_RAW_DIR = DATA_RAW_DIR / "ieee_cis"
IEEE_TRANSACTION_FILE = IEEE_RAW_DIR / "train_transaction.csv"
IEEE_IDENTITY_FILE = IEEE_RAW_DIR / "train_identity.csv"
IEEE_MODEL_ARTIFACT_FILE = MODELS_DIR / "model_ieee.pkl"

# --- Data split ---
# Time-aware split: training data precedes test data (no shuffling), to
# mirror a real production deployment where the model only ever sees past
# transactions at training time.
TEST_SIZE = 0.2

# --- Imbalance handling ---
RANDOM_STATE = 42

# --- Model hyperparameters (XGBoost) ---
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "eval_metric": "aucpr",
}

# --- Cost-sensitive threshold selection ---
# Illustrative costs used to pick the operating threshold that minimises
# expected cost (see src/models/evaluate.py). Values are placeholders to be
# justified/tuned against the dataset and documented in the README.
COST_FALSE_NEGATIVE = 100.0  # average fraud loss when a fraud is missed
COST_FALSE_POSITIVE = 5.0  # customer friction / investigation cost
# Cost-minimising threshold for the class-weighted policy (chosen as primary
# in Phase 3 — see README "Phase 3" for the shadow-mode A/B test reasoning).
DECISION_THRESHOLD = 0.29

# --- Drift monitoring ---
DRIFT_PSI_THRESHOLD = 0.2

# --- Shadow-mode A/B test (SMOTE vs. class-weighting) ---
# Offline statistical comparison, not a live experiment — see README
# "Why a shadow-mode A/B test" for what this does and doesn't demonstrate.
AB_TEST_BOOTSTRAP_SAMPLES = 10_000
AB_TEST_CONFIDENCE_LEVEL = 0.95
