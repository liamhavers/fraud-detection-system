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
# Both imbalance-handling variants are kept around (not just the winner) so
# the Phase 3 shadow-mode A/B test has two real trained policies to compare.
MODEL_ARTIFACT_CLASS_WEIGHTED_FILE = MODELS_DIR / "model_class_weighted.pkl"
MODEL_ARTIFACT_SMOTE_FILE = MODELS_DIR / "model_smote.pkl"
# Phase 7 calibration control: same XGB_PARAMS, no reweighting or
# resampling. Log loss is a proper scoring rule, so this model is expected
# to be roughly calibrated without any correction.
MODEL_ARTIFACT_UNWEIGHTED_FILE = MODELS_DIR / "model_unweighted.pkl"
# Class-weighting was kept as the primary served policy after Phase 3's
# shadow-mode A/B test came back statistically inconclusive (95% CI on the
# cost difference spanned zero) — chosen on point-estimate cost/PR-AUC and
# simplicity grounds, not because the test proved it superior. See README
# "Phase 3" for the full reasoning.
MODEL_ARTIFACT_FILE = MODEL_ARTIFACT_CLASS_WEIGHTED_FILE

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
# Validation slice (fraction of the full dataset) for model selection,
# threshold search and calibrator fitting, so test is touched once, for
# final reporting only. It is carved from the end of what used to be
# training data, giving a 60 / 20 / 20 split in which the test slice (the
# last 20%) is row-for-row identical to the old 80 / 20 split's test slice,
# so old and new test numbers are directly comparable.
VAL_SIZE = 0.2

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

# --- Probability calibration (Phase 7) ---
# Reliability-table bin edges on predicted probability. At a ~0.1-0.2% base
# rate, equal-width or quantile bins put nearly every fraud in one bin, so
# the low end is log-spaced (where the ~0.048 decision threshold sits). The
# middle is merged into two wide bins (0.01-0.1, 0.1-0.5) because the
# validation slice has only 57 frauds and finer middle bins held 5-12
# transactions each.
CALIBRATION_BIN_EDGES = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0)
CALIBRATION_CONFIDENCE_LEVEL = 0.95  # Wilson intervals on per-bin fraud rates
# Threshold grid for calibrated scores: the cost-optimal threshold for a
# calibrated model is ~0.048, so a fine log-spaced grid is needed rather
# than the 0.01-0.99 linear grid used for raw (inflated) scores.
FINE_THRESHOLD_GRID_MIN = 1e-4
FINE_THRESHOLD_GRID_MAX = 0.99
FINE_THRESHOLD_GRID_SIZE = 500
# Calibration method used downstream (A/B test). Chosen by 5-fold
# cross-validated log loss on validation in src/models/calibration_report.py,
# which picked Platt over isotonic for all three XGBoost variants (57
# validation frauds are too few for isotonic's step function).
CALIBRATION_METHOD = "platt"

# --- Drift monitoring ---
DRIFT_PSI_THRESHOLD = 0.2

# --- Shadow-mode A/B test (SMOTE vs. class-weighting) ---
# Offline statistical comparison, not a live experiment — see README
# "Why a shadow-mode A/B test" for what this does and doesn't demonstrate.
AB_TEST_BOOTSTRAP_SAMPLES = 10_000
AB_TEST_CONFIDENCE_LEVEL = 0.95
