"""Probability calibration (Platt, isotonic) and calibration metrics.

Why this exists: PR-AUC only measures *ranking* (are frauds scored above
legitimate transactions?). It says nothing about whether a score of 0.05
means a 5% chance of fraud. Class weighting and SMOTE both train the model
on a much higher fraud rate than the real ~0.1-0.2%, so their scores are not
probabilities. Calibration fixes the *level* without changing the ranking,
and once scores are real probabilities the cost-optimal threshold follows
directly from decision theory (`analytic_threshold`).

Calibrators are fitted on the validation slice only; test is for reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_CONFIDENCE_LEVEL,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MODELS_DIR,
    RANDOM_STATE,
)

_EPS = 1e-15


# --- Helpers ---------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def _fit_unpenalised_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y ~ sigmoid(slope * x + intercept) by maximum likelihood.

    `C=np.inf` means no regularisation (scikit-learn 1.9 deprecates
    `penalty=None` in favour of this). The default L2 penalty would shrink
    the slope towards 0, biasing exactly the parameters we want to read.
    """
    model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
    model.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def model_log_odds(model: XGBClassifier | object, X: pd.DataFrame) -> np.ndarray:
    """The model's raw log-odds score (before the sigmoid).

    XGBoost exposes its margin directly, which is cleaner than taking the
    logit of a probability that has been clipped near 0 or 1. Other models
    (e.g. the logistic-regression baseline pipeline) use decision_function.
    """
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True)
    return model.decision_function(X)


# --- Calibrators -----------------------------------------------------------


def fit_platt(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: fit logit(p_cal) = a * score + b. Returns (a, b).

    Reading the parameters: if a model was trained on an inflated fraud rate
    but otherwise ranks well, its log-odds are shifted by a constant, so
    a ≈ 1 and b ≈ -(the shift). a < 1 means the raw scores were too spread
    out (overconfident), a > 1 too compressed.
    """
    return _fit_unpenalised_logistic(score, y)


def apply_platt(score: np.ndarray, a: float, b: float) -> np.ndarray:
    """Calibrated probability from a log-odds score."""
    return _sigmoid(a * np.asarray(score, dtype=float) + b)


def fit_isotonic(proba: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Isotonic regression: the best-fitting non-decreasing step function.

    More flexible than Platt (no assumed shape), so it needs more positives
    to avoid fitting noise: each step is an average over a run of points.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(np.asarray(proba, dtype=float), np.asarray(y))
    return iso


@dataclass
class Calibrator:
    """A fitted calibrator for one model: Platt (a, b) or isotonic."""

    method: Literal["platt", "isotonic"]
    platt_a: float | None = None
    platt_b: float | None = None
    isotonic: IsotonicRegression | None = None

    @classmethod
    def fit(
        cls,
        method: Literal["platt", "isotonic"],
        model: XGBClassifier | object,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "Calibrator":
        """Fit on the given (validation) data."""
        y = np.asarray(y)
        if method == "platt":
            a, b = fit_platt(model_log_odds(model, X), y)
            return cls(method="platt", platt_a=a, platt_b=b)
        if method == "isotonic":
            return cls(method="isotonic", isotonic=fit_isotonic(model.predict_proba(X)[:, 1], y))
        raise ValueError(f"unknown calibration method: {method!r}")

    def predict_proba(self, model: XGBClassifier | object, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of fraud for each row of X."""
        if self.method == "platt":
            return apply_platt(model_log_odds(model, X), self.platt_a, self.platt_b)
        return self.isotonic.predict(model.predict_proba(X)[:, 1])


def cross_validated_log_loss(
    method: Literal["platt", "isotonic"],
    model: XGBClassifier | object,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> float:
    """Out-of-fold log loss of a calibration method, within one data slice.

    Used to choose between Platt and isotonic on validation without touching
    test. Scoring a calibrator on the same rows it was fitted on would always
    favour the more flexible isotonic fit; here each fold's calibrator is
    scored only on rows it did not see. Folds are stratified so each holds a
    share of the (few) frauds.
    """
    y = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold = np.empty(len(y), dtype=float)
    for fit_idx, score_idx in folds.split(X, y):
        calibrator = Calibrator.fit(method, model, X.iloc[fit_idx], y[fit_idx])
        out_of_fold[score_idx] = calibrator.predict_proba(model, X.iloc[score_idx])
    return log_loss_score(y, out_of_fold)


def calibrator_artifact_path(model_name: str, method: str) -> Path:
    """Where a fitted calibrator is saved, e.g. models/calibrator_smote_platt.pkl."""
    return MODELS_DIR / f"calibrator_{model_name}_{method}.pkl"


# --- Metrics ---------------------------------------------------------------


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome."""
    return float(brier_score_loss(np.asarray(y), np.asarray(p)))


def brier_skill_score(y: np.ndarray, p: np.ndarray) -> float:
    """1 - Brier / Brier of always predicting the observed base rate π.

    The reference Brier score is π(1 - π). At a 0.1% base rate every raw
    Brier score is tiny, so this puts it on a readable scale: 0 means no
    better than the base rate, 1 is perfect, negative is worse than useless.
    """
    pi = float(np.mean(y))
    return 1 - brier_score(y, p) / (pi * (1 - pi))


def log_loss_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log-likelihood; p clipped to avoid log(0)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_in_the_large(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Mean predicted vs observed rate, and observed / expected (O/E).

    O/E < 1: the model predicts more fraud than happens (inflated scores).
    O/E > 1: it predicts less.
    """
    mean_predicted = float(np.mean(p))
    observed_rate = float(np.mean(y))
    return {
        "mean_predicted": mean_predicted,
        "observed_rate": observed_rate,
        "o_to_e": observed_rate / mean_predicted,
    }


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Logistic regression of y on logit(p): returns slope and intercept.

    A perfectly calibrated model gives slope 1, intercept 0. Slope < 1 means
    overconfident (predictions too extreme), > 1 underconfident. An intercept
    away from 0 means the whole curve is shifted.
    """
    slope, intercept = _fit_unpenalised_logistic(_logit(p), y)
    return {"slope": slope, "intercept": intercept}


def wilson_interval(
    successes: int, n: int, confidence: float = CALIBRATION_CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook p ± z·sqrt(p(1-p)/n) interval, which
    collapses to zero width at 0 or n successes and goes outside [0, 1]
    for rare events, which is exactly the regime of a fraud bin.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z = float(norm.ppf(0.5 + confidence / 2))
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    bin_edges: tuple[float, ...] | list[float],
    confidence: float = CALIBRATION_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Per-bin n, frauds, mean predicted, observed rate and its Wilson interval.

    Bins are [lower, upper), with the last bin closed at 1. Empty bins are
    kept (n = 0, rates NaN) so the table always has one row per bin.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    bin_index = np.digitize(p, edges[1:-1], right=False)

    rows = []
    for i in range(len(edges) - 1):
        in_bin = bin_index == i
        n = int(in_bin.sum())
        frauds = int(y[in_bin].sum())
        lower, upper = wilson_interval(frauds, n, confidence) if n else (np.nan, np.nan)
        rows.append(
            {
                "bin_lower": edges[i],
                "bin_upper": edges[i + 1],
                "n": n,
                "frauds": frauds,
                "mean_predicted": float(p[in_bin].mean()) if n else np.nan,
                "observed_rate": frauds / n if n else np.nan,
                "observed_lower": lower,
                "observed_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """n-weighted mean |mean predicted - observed rate| across bins.

    Secondary metric only: its value depends on the choice of bin edges.
    """
    filled = table[table["n"] > 0]
    gaps = (filled["mean_predicted"] - filled["observed_rate"]).abs()
    return float((filled["n"] * gaps).sum() / filled["n"].sum())


# --- Decision threshold ----------------------------------------------------


def analytic_threshold(
    fn_cost: float = COST_FALSE_NEGATIVE, fp_cost: float = COST_FALSE_POSITIVE
) -> float:
    """Cost-optimal threshold for a *calibrated* probability.

    Flag when the expected cost of not flagging, p * fn_cost, exceeds the
    expected cost of flagging, (1 - p) * fp_cost, i.e. when
    p > fp_cost / (fp_cost + fn_cost). Only valid if p is a real probability.
    """
    return fp_cost / (fp_cost + fn_cost)
