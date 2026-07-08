"""Evaluation: PR-AUC, confusion matrices, and cost-sensitive threshold selection.

Why cost-sensitive thresholding: with ~0.17% positive class, accuracy is
meaningless and the default 0.5 probability threshold is arbitrary. Picking
the threshold that minimises expected business cost (false negatives ~
missed fraud losses, false positives ~ customer friction/investigation cost)
demonstrates a business-aware operating point rather than pure metric-chasing.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


def pr_auc_score(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Primary metric: precision-recall AUC."""
    return float(average_precision_score(y_true, y_proba))


def expected_cost(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Total expected cost given false negative / false positive costs."""
    raise NotImplementedError


def select_cost_minimising_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    fn_cost: float = COST_FALSE_NEGATIVE,
    fp_cost: float = COST_FALSE_POSITIVE,
) -> float:
    """Sweep candidate thresholds and return the one minimising expected cost."""
    raise NotImplementedError
