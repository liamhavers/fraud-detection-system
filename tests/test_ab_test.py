"""Sanity checks for the shadow-mode A/B comparison logic on synthetic data."""

import numpy as np
import pytest

from src.models.ab_test import (
    bootstrap_cost_difference,
    compare_policies,
    paired_bootstrap_cost_difference,
    per_transaction_cost,
)


def test_bootstrap_detects_known_cost_difference() -> None:
    rng = np.random.default_rng(42)
    cost_a = rng.normal(loc=10.0, scale=1.0, size=1000)
    cost_b = rng.normal(loc=15.0, scale=1.0, size=1000)

    result = bootstrap_cost_difference(cost_a, cost_b)

    assert result["point_estimate"] < 0
    assert result["ci_upper"] < 0


def test_bootstrap_ci_contains_zero_when_no_true_difference() -> None:
    rng = np.random.default_rng(42)
    cost_a = rng.normal(loc=10.0, scale=1.0, size=1000)
    cost_b = rng.normal(loc=10.0, scale=1.0, size=1000)

    result = bootstrap_cost_difference(cost_a, cost_b)

    assert result["ci_lower"] < 0 < result["ci_upper"]


def test_per_transaction_cost_charges_fn_and_fp_only() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])  # correct, FP, correct, FN

    cost = per_transaction_cost(y_true, y_pred, fn_cost=100.0, fp_cost=5.0)

    assert cost.tolist() == [0.0, 5.0, 0.0, 100.0]


def test_compare_policies_detects_known_cost_difference() -> None:
    n_fraud = 100
    y_true = np.zeros(2000, dtype=int)
    y_true[:n_fraud] = 1

    # Policy A misses far more fraud than policy B -> policy A costs more.
    y_pred_a = y_true.copy()
    y_pred_a[:60] = 0
    y_pred_b = y_true.copy()
    y_pred_b[:10] = 0

    result = compare_policies(y_true, y_pred_a, y_pred_b, label_a="a", label_b="b")

    assert result["point_estimate"] > 0
    assert result["ci_lower"] > 0
    assert result["statistically_significant"] is True


def test_compare_policies_identical_predictions_are_not_significant() -> None:
    y_true = np.array([0, 0, 1, 1] * 500)
    y_pred = np.array([0, 0, 1, 0] * 500)

    result = compare_policies(y_true, y_pred, y_pred, label_a="a", label_b="b")

    assert result["point_estimate"] == 0.0
    assert result["statistically_significant"] is False


def _correlated_costs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two policies that agree almost everywhere; A avoids 200 of B's £5 FPs.

    Known truth: A is cheaper by exactly 200 * 5 / n per transaction. The
    occasional £100 misses are shared by both policies, so they add a lot of
    noise to each policy's cost but none to the difference.
    """
    rng = np.random.default_rng(seed)
    n = 20_000
    cost_b = rng.choice([0.0, 5.0, 100.0], size=n, p=[0.97, 0.02, 0.01])
    cost_a = cost_b.copy()
    fp_rows = np.flatnonzero(cost_b == 5.0)[:200]
    cost_a[fp_rows] = 0.0
    return cost_a, cost_b


def test_paired_bootstrap_detects_small_effect_that_unpaired_misses() -> None:
    cost_a, cost_b = _correlated_costs()

    paired = paired_bootstrap_cost_difference(cost_a, cost_b, n_bootstrap=2_000)
    unpaired = bootstrap_cost_difference(cost_a, cost_b, n_bootstrap=2_000)

    assert paired["point_estimate"] == pytest.approx(-200 * 5 / 20_000)
    assert paired["ci_upper"] < 0  # paired design: clearly distinguishable
    assert unpaired["ci_lower"] < 0 < unpaired["ci_upper"]  # unpaired: lost in noise
    assert paired["n_discordant"] == 200


def test_paired_bootstrap_of_identical_policies_is_exactly_zero() -> None:
    cost = _correlated_costs()[1]

    result = paired_bootstrap_cost_difference(cost, cost, n_bootstrap=500)

    assert result["ci_lower"] == result["ci_upper"] == result["point_estimate"] == 0.0


def test_paired_bootstrap_requires_matching_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_cost_difference(np.zeros(3), np.zeros(4))
