"""Tests for src/models/evaluate.py."""

import numpy as np

from src.models.evaluate import pr_auc_score


def test_pr_auc_score_is_perfect_for_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    assert pr_auc_score(y_true, y_proba) == 1.0


def test_pr_auc_score_penalises_poor_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    good_proba = np.array([0.1, 0.2, 0.8, 0.9])
    bad_proba = np.array([0.9, 0.8, 0.2, 0.1])

    assert pr_auc_score(y_true, bad_proba) < pr_auc_score(y_true, good_proba)
