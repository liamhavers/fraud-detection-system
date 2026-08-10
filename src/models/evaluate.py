"""Evaluation: PR-AUC, confusion matrices, and cost-sensitive threshold selection.

Why cost-sensitive thresholding: with ~0.17% positive class, accuracy is
meaningless and the default 0.5 probability threshold is arbitrary. Picking
the threshold that minimises expected business cost (false negatives ~
missed fraud losses, false positives ~ customer friction/investigation cost)
demonstrates a business-aware operating point rather than pure metric-chasing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from src.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

# Categorical colors reused from the EDA notebooks for consistency across
# the project's visuals: blue = legitimate, red = fraud.
_POLICY_COLORS = {
    "class_weighted": "#2a78d6",
    "smote": "#e34948",
}
_SURFACE_COLOR = "#fcfcfb"
_GRID_COLOR = "#e1e0d9"


def pr_auc_score(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Primary metric: precision-recall AUC."""
    return float(average_precision_score(y_true, y_proba))


def expected_cost(
    y_true: pd.Series,
    y_pred: np.ndarray,
    fn_cost: float = COST_FALSE_NEGATIVE,
    fp_cost: float = COST_FALSE_POSITIVE,
) -> float:
    """Total expected cost given false negative / false positive costs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    n_false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    return float(n_false_negatives * fn_cost + n_false_positives * fp_cost)


def confusion_counts(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """TP/FP/TN/FN counts at the given predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "true_positive": int(np.sum((y_true == 1) & (y_pred == 1))),
        "false_positive": int(np.sum((y_true == 0) & (y_pred == 1))),
        "true_negative": int(np.sum((y_true == 0) & (y_pred == 0))),
        "false_negative": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def select_cost_minimising_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    fn_cost: float = COST_FALSE_NEGATIVE,
    fp_cost: float = COST_FALSE_POSITIVE,
) -> float:
    """Sweep candidate thresholds and return the one minimising expected cost."""
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    costs = [
        expected_cost(y_true, (y_proba >= t).astype(int), fn_cost, fp_cost)
        for t in candidate_thresholds
    ]
    return float(candidate_thresholds[int(np.argmin(costs))])


def plot_precision_recall_curves(
    y_true: pd.Series, proba_by_policy: dict[str, np.ndarray], save_path: Path
) -> None:
    """Overlay PR curves for one or more policies and save to disk."""
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=_SURFACE_COLOR)
    ax.set_facecolor(_SURFACE_COLOR)

    for label, proba in proba_by_policy.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(
            recall,
            precision,
            color=_POLICY_COLORS.get(label, "#898781"),
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Kaggle Credit Card Fraud")
    ax.grid(color=_GRID_COLOR, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, save_path: Path, title: str = "Confusion Matrix"
) -> None:
    """Render a 2x2 confusion matrix heatmap and save to disk."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4, 4), facecolor=_SURFACE_COLOR)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
