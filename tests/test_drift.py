"""Tests for src/monitoring/drift.py."""

import numpy as np
import polars as pl

from src.monitoring.drift import check_drift, population_stability_index


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
    reference_lf = pl.LazyFrame(
        {"stable": rng.normal(size=2000), "shifted": rng.normal(loc=0.0, size=2000)}
    )
    current_lf = pl.LazyFrame(
        {"stable": rng.normal(size=2000), "shifted": rng.normal(loc=5.0, size=2000)}
    )

    report = check_drift(reference_lf, current_lf, threshold=0.2)

    drifted_features = set(report.filter(pl.col("drifted"))["feature"])
    assert drifted_features == {"shifted"}


def test_check_drift_ignores_nulls_and_non_numeric_columns() -> None:
    reference_lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, None], "label": list("abcde")})
    current_lf = pl.LazyFrame({"x": [None, 1.0, 2.0, 3.0, 4.0], "label": list("vwxyz")})

    report = check_drift(reference_lf, current_lf)

    assert report["feature"].to_list() == ["x"]
    assert report["psi"].item() == 0.0


def test_psi_floors_empty_current_bins_instead_of_returning_inf() -> None:
    reference = np.arange(100, dtype=float)
    current = np.full(100, 1000.0)  # all mass in the last bin

    psi = population_stability_index(reference, current)

    assert np.isfinite(psi) and psi > 1.0
