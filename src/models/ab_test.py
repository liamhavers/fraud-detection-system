"""Offline shadow-mode statistical comparison between two trained policies.

This replays historical held-out data through two competing policies
(SMOTE-resampled model vs. class-weighted model) and compares their expected
cost using bootstrapped confidence intervals, rather than picking a winner by
eyeballing metrics.

This is deliberately an offline simulation for building statistical
intuition — not a live, traffic-splitting production experiment. It has no
real traffic split and no protection against novelty or seasonality effects
a real online test would need to control for. See README "Why a shadow-mode
A/B test" for the full framing.
"""

import numpy as np
import pandas as pd

from src.config import AB_TEST_BOOTSTRAP_SAMPLES, AB_TEST_CONFIDENCE_LEVEL, RANDOM_STATE


def per_transaction_cost(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Cost incurred by each transaction under a given policy's predictions."""
    raise NotImplementedError


def bootstrap_cost_difference(
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    n_bootstrap: int = AB_TEST_BOOTSTRAP_SAMPLES,
    confidence_level: float = AB_TEST_CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Bootstrap the mean cost difference (A - B) and return point estimate + CI.

    A percentile bootstrap: resample each cost array with replacement
    n_bootstrap times, take the mean difference each time, then read off the
    confidence interval from the resulting distribution's percentiles.
    """
    rng = np.random.default_rng(random_state)
    cost_a, cost_b = np.asarray(cost_a), np.asarray(cost_b)

    idx_a = rng.integers(0, len(cost_a), size=(n_bootstrap, len(cost_a)))
    idx_b = rng.integers(0, len(cost_b), size=(n_bootstrap, len(cost_b)))
    diffs = cost_a[idx_a].mean(axis=1) - cost_b[idx_b].mean(axis=1)

    alpha = 1 - confidence_level
    ci_lower, ci_upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": cost_a.mean() - cost_b.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
    }


def compare_policies(
    y_true: pd.Series,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    label_a: str = "smote",
    label_b: str = "class_weighted",
) -> dict:
    """Full shadow-mode comparison: expected cost per policy, bootstrapped CI on
    the difference, and whether it's statistically distinguishable from noise.
    """
    raise NotImplementedError
