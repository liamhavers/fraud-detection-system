"""Tests for src/models/calibrate.py, on synthetic data with a known truth.

The core simulation: draw a true log-odds z at a low base rate, set
p_true = sigmoid(z) and y ~ Bernoulli(p_true). Then distort the scores in a
known way and check that the metrics and calibrators detect and undo it.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
from src.models.calibrate import (
    Calibrator,
    analytic_threshold,
    apply_platt,
    brier_skill_score,
    calibration_in_the_large,
    calibration_slope_intercept,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    log_loss_score,
    model_log_odds,
    reliability_table,
    wilson_interval,
)
from src.models.evaluate import (
    fine_threshold_grid,
    plot_reliability_diagram,
    pr_auc_score,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _simulate(n: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """True log-odds z (base rate ~1-2%) and outcomes y ~ Bernoulli(sigmoid(z))."""
    rng = np.random.default_rng(seed)
    z = rng.normal(loc=-5.0, scale=1.5, size=n)
    y = rng.binomial(1, _sigmoid(z))
    return z, y


# --- Platt scaling ---------------------------------------------------------


def test_platt_recovers_known_odds_inflation() -> None:
    # Inflating the odds by w (as class weighting by w does in theory) shifts
    # the log-odds by log(w). Platt should learn a ≈ 1, b ≈ -log(w).
    z, y = _simulate()
    w = 50.0
    inflated_score = z + np.log(w)

    a, b = fit_platt(inflated_score, y)

    assert a == pytest.approx(1.0, abs=0.05)
    assert b == pytest.approx(-np.log(w), abs=0.2)
    raw = calibration_in_the_large(y, _sigmoid(inflated_score))
    calibrated = calibration_in_the_large(y, apply_platt(inflated_score, a, b))
    assert raw["o_to_e"] < 0.2  # badly inflated before
    assert calibrated["o_to_e"] == pytest.approx(1.0, abs=0.05)


def test_platt_leaves_pr_auc_exactly_unchanged() -> None:
    # Strictly increasing transform (a > 0) => same ranking => same PR-AUC.
    z, y = _simulate(n=50_000)
    score = z + np.log(20.0)
    a, b = fit_platt(score, y)

    assert a > 0
    assert pr_auc_score(y, apply_platt(score, a, b)) == pr_auc_score(y, _sigmoid(score))


# --- Isotonic regression ---------------------------------------------------


def test_isotonic_is_monotone_and_bounded() -> None:
    z, y = _simulate(n=50_000)
    iso = fit_isotonic(_sigmoid(z + np.log(20.0)), y)

    grid = np.linspace(0, 1, 1_001)
    out = iso.predict(grid)

    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 0 and out.max() <= 1


# --- Calibrator wrapper ----------------------------------------------------


def test_model_log_odds_matches_xgboost_probability() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.normal(size=500)})
    y = (X["f"] + rng.normal(size=500) > 1).astype(int)
    model = XGBClassifier(n_estimators=10, max_depth=2).fit(X, y)

    np.testing.assert_allclose(
        _sigmoid(model_log_odds(model, X)), model.predict_proba(X)[:, 1], rtol=1e-5
    )


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_fixes_a_class_weighted_model(method: str) -> None:
    # A class-weighted logistic regression is inflated; calibrating on
    # held-out data should bring O/E back near 1.
    rng = np.random.default_rng(1)
    n = 60_000
    X = pd.DataFrame({"f": rng.normal(size=n)})
    y = rng.binomial(1, _sigmoid(-5 + 1.5 * X["f"].to_numpy()))
    model = LogisticRegression(class_weight="balanced").fit(X[: n // 2], y[: n // 2])
    X_val, y_val = X[n // 2 :], y[n // 2 :]

    calibrator = Calibrator.fit(method, model, X_val, y_val)
    p_cal = calibrator.predict_proba(model, X_val)

    assert calibration_in_the_large(y_val, model.predict_proba(X_val)[:, 1])["o_to_e"] < 0.2
    assert calibration_in_the_large(y_val, p_cal)["o_to_e"] == pytest.approx(1.0, abs=0.05)


def test_calibrator_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        Calibrator.fit("beta", None, pd.DataFrame(), pd.Series(dtype=int))


# --- Calibration metrics ---------------------------------------------------


def test_perfectly_calibrated_input_has_slope_one_intercept_zero() -> None:
    z, y = _simulate()

    result = calibration_slope_intercept(y, _sigmoid(z))

    assert result["slope"] == pytest.approx(1.0, abs=0.05)
    assert result["intercept"] == pytest.approx(0.0, abs=0.25)
    assert calibration_in_the_large(y, _sigmoid(z))["o_to_e"] == pytest.approx(1.0, abs=0.05)


def test_overconfident_input_has_slope_half() -> None:
    # Doubling the log-odds makes predictions too extreme; the slope of y on
    # logit(p) = 2z is then 1/2.
    z, y = _simulate()

    result = calibration_slope_intercept(y, _sigmoid(2 * z))

    assert result["slope"] == pytest.approx(0.5, abs=0.03)


def test_brier_skill_score_is_zero_at_the_base_rate() -> None:
    y = np.array([0] * 990 + [1] * 10)

    assert brier_skill_score(y, np.full(len(y), y.mean())) == pytest.approx(0.0, abs=1e-12)
    assert brier_skill_score(y, y.astype(float)) == pytest.approx(1.0)


def test_log_loss_handles_hard_zeros_and_ones() -> None:
    y = np.array([0, 1])

    assert log_loss_score(y, np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(log_loss_score(y, np.array([1.0, 0.0])))


# --- Wilson interval and reliability table ---------------------------------


def test_wilson_interval_has_roughly_nominal_coverage() -> None:
    rng = np.random.default_rng(2)
    true_rate, n, draws = 0.03, 200, 4_000
    successes = rng.binomial(n, true_rate, size=draws)

    covered = [lo <= true_rate <= hi for lo, hi in (wilson_interval(k, n) for k in successes)]

    assert 0.92 <= np.mean(covered) <= 0.98


def test_wilson_interval_handles_zero_and_all_successes() -> None:
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1

    lo, hi = wilson_interval(50, 50)
    assert 0.9 < lo < 1 and hi == 1.0

    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_reliability_table_counts_and_ece() -> None:
    p = np.array([0.0, 0.05, 0.2, 0.2, 0.6, 1.0])
    y = np.array([0, 0, 1, 0, 1, 1])

    table = reliability_table(y, p, bin_edges=[0, 0.1, 0.5, 1.0])

    assert table["n"].tolist() == [2, 2, 2]  # 1.0 lands in the last bin
    assert table["frauds"].tolist() == [0, 1, 2]
    assert table["observed_rate"].tolist() == [0.0, 0.5, 1.0]
    # gaps: |0.025 - 0| , |0.2 - 0.5| , |0.8 - 1| -> weighted mean 0.175
    assert expected_calibration_error(table) == pytest.approx(0.175)


def test_reliability_table_keeps_empty_bins() -> None:
    table = reliability_table(np.array([0, 1]), np.array([0.01, 0.02]), bin_edges=[0, 0.5, 1])

    assert table["n"].tolist() == [2, 0]
    assert np.isnan(table.loc[1, "observed_rate"])
    assert expected_calibration_error(table) == pytest.approx(0.485)


def test_plot_reliability_diagram_writes_file(tmp_path) -> None:
    z, y = _simulate(n=20_000)
    edges = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    tables = {
        "unweighted": reliability_table(y, _sigmoid(z), edges),
        "smote · platt": reliability_table(y, _sigmoid(z + 3), edges),
    }
    path = tmp_path / "reliability.png"

    plot_reliability_diagram(tables, path)

    assert path.exists()


# --- Thresholds ------------------------------------------------------------


def test_analytic_threshold_from_config_costs() -> None:
    assert COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE) == 5 / 105
    assert analytic_threshold() == pytest.approx(5 / 105)


def test_fine_threshold_grid_is_log_spaced_and_covers_the_optimum() -> None:
    grid = fine_threshold_grid()

    assert grid.min() < analytic_threshold() < grid.max()
    assert np.allclose(np.diff(np.log(grid)), np.log(grid[1] / grid[0]))
