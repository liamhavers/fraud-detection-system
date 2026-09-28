"""Tests for src/pandas_reference/drift.py."""

import numpy as np
import pandas as pd

from src.pandas_reference.drift import check_drift, population_stability_index


def test_psi_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=5000)
    current = rng.normal(size=5000)

    assert population_stability_index(reference, current) < 0.05


def test_psi_is_large_for_a_shifted_distribution() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0.0, size=5000)
    current = rng.normal(loc=5.0, size=5000)

    assert population_stability_index(reference, current) > 0.2


def test_psi_handles_near_constant_reference() -> None:
    reference = np.ones(100)
    current = np.ones(100) * 2

    assert population_stability_index(reference, current) == 0.0


def test_check_drift_flags_only_the_shifted_feature() -> None:
    rng = np.random.default_rng(42)
    reference_df = pd.DataFrame(
        {"stable": rng.normal(size=2000), "shifted": rng.normal(loc=0.0, size=2000)}
    )
    current_df = pd.DataFrame(
        {"stable": rng.normal(size=2000), "shifted": rng.normal(loc=5.0, size=2000)}
    )

    report = check_drift(reference_df, current_df, threshold=0.2)

    drifted_features = set(report.loc[report["drifted"], "feature"])
    assert drifted_features == {"shifted"}
