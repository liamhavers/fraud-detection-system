"""Tests for src/models/evaluate.py."""

import numpy as np

from src.models.evaluate import (
    confusion_counts,
    expected_cost,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    pr_auc_score,
    select_cost_minimising_threshold,
)


def test_pr_auc_score_is_perfect_for_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    assert pr_auc_score(y_true, y_proba) == 1.0


def test_pr_auc_score_penalises_poor_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    good_proba = np.array([0.1, 0.2, 0.8, 0.9])
    bad_proba = np.array([0.9, 0.8, 0.2, 0.1])

    assert pr_auc_score(y_true, bad_proba) < pr_auc_score(y_true, good_proba)


def test_confusion_counts_matches_known_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    assert confusion_counts(y_true, y_pred) == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }


def test_expected_cost_weights_false_negatives_and_positives() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])  # 1 FP, 1 FN

    assert expected_cost(y_true, y_pred, fn_cost=100.0, fp_cost=5.0) == 105.0


def test_select_cost_minimising_threshold_prefers_cheap_false_positives() -> None:
    # Fraud (class 1) scores high; with FN much costlier than FP, the optimal
    # threshold should sit low enough to catch it even at the cost of a few FPs.
    y_true = np.array([0, 0, 0, 1])
    y_proba = np.array([0.2, 0.3, 0.4, 0.6])

    threshold = select_cost_minimising_threshold(
        y_true, y_proba, fn_cost=100.0, fp_cost=1.0
    )
    y_pred = (y_proba >= threshold).astype(int)

    assert y_pred[-1] == 1  # the fraud case is caught


def test_plot_functions_write_files(tmp_path) -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = np.array([0, 0, 1, 1])

    pr_path = tmp_path / "pr_curve.png"
    cm_path = tmp_path / "confusion_matrix.png"

    plot_precision_recall_curves(y_true, {"policy_a": y_proba}, pr_path)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    assert pr_path.exists()
    assert cm_path.exists()
