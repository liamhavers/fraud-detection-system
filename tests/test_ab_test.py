"""Sanity checks for the shadow-mode A/B comparison logic on synthetic data."""

import numpy as np

from src.models.ab_test import bootstrap_cost_difference


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
