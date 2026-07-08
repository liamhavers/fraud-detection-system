"""Single source of truth for paths, thresholds, and model hyperparameters."""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

RAW_DATA_FILE = DATA_RAW_DIR / "creditcard.csv"
TRAIN_DATA_FILE = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_FILE = DATA_PROCESSED_DIR / "test.csv"
MODEL_ARTIFACT_FILE = MODELS_DIR / "model.pkl"

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
DECISION_THRESHOLD = 0.5  # overwritten once cost-sensitive analysis is done

# --- Drift monitoring ---
DRIFT_PSI_THRESHOLD = 0.2
