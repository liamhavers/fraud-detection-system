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

from src.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FINE_THRESHOLD_GRID_MAX,
    FINE_THRESHOLD_GRID_MIN,
    FINE_THRESHOLD_GRID_SIZE,
)

# Categorical colors reused from the EDA notebooks for consistency across
# the project's visuals: blue = legitimate, red = fraud. Aqua is the next
# unused categorical slot, for the Phase 7 unweighted control. Markers give
# each policy a second, non-colour encoding.
_POLICY_COLORS = {
    "class_weighted": "#2a78d6",
    "smote": "#e34948",
    "unweighted": "#1baf7a",
}
_POLICY_MARKERS = {"class_weighted": "o", "smote": "s", "unweighted": "^"}
_NEUTRAL_COLOR = "#898781"
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
    thresholds: np.ndarray | None = None,
) -> float:
    """Sweep candidate thresholds and return the one minimising expected cost.

    The default grid (0.01 to 0.99 in steps of 0.01) suits raw, inflated
    scores. For calibrated probabilities, whose optimum sits near 0.048,
    pass `fine_threshold_grid()` instead.
    """
    candidate_thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    costs = [
        expected_cost(y_true, (y_proba >= t).astype(int), fn_cost, fp_cost)
        for t in candidate_thresholds
    ]
    return float(candidate_thresholds[int(np.argmin(costs))])


def fine_threshold_grid() -> np.ndarray:
    """Log-spaced threshold grid for calibrated probabilities."""
    return np.geomspace(
        FINE_THRESHOLD_GRID_MIN, FINE_THRESHOLD_GRID_MAX, FINE_THRESHOLD_GRID_SIZE
    )


def _policy_key(label: str) -> str | None:
    """Match a label like "smote · platt" to its policy for colour/marker."""
    return next((key for key in _POLICY_COLORS if label.startswith(key)), None)


def plot_reliability_diagram(
    tables_by_label: dict[str, pd.DataFrame], save_path: Path, title: str = "Reliability diagram"
) -> None:
    """Observed fraud rate vs mean predicted probability, per bin, on log axes.

    Takes `reliability_table` outputs. Points on the dashed diagonal are
    perfectly calibrated; below it the model over-predicts fraud. Error bars
    are Wilson intervals. Bins with zero frauds can't sit on a log y-axis, so
    they are drawn as hollow markers on the bottom edge (with their Wilson
    upper bound) rather than dropped. The lower panel shows n per bin.
    """
    filled = [t[t["n"] > 0] for t in tables_by_label.values()]
    positives = np.concatenate(
        [t["mean_predicted"].to_numpy() for t in filled]
        + [t["observed_lower"].to_numpy() for t in filled]
    )
    axis_min = max(1e-6, 10 ** np.floor(np.log10(positives[positives > 0].min())))

    fig, (ax, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(6, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor=_SURFACE_COLOR,
    )
    for a in (ax, ax_hist):
        a.set_facecolor(_SURFACE_COLOR)
        a.set_xscale("log")
        a.grid(color=_GRID_COLOR, linewidth=0.5)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax.plot([axis_min, 1], [axis_min, 1], linestyle="--", color=_NEUTRAL_COLOR, linewidth=1)

    for label, table in tables_by_label.items():
        key = _policy_key(label)
        color = _POLICY_COLORS.get(key, _NEUTRAL_COLOR)
        marker = _POLICY_MARKERS.get(key, "D")
        table = table[table["n"] > 0]
        has_fraud = table[table["frauds"] > 0]
        no_fraud = table[table["frauds"] == 0]

        ax.errorbar(
            has_fraud["mean_predicted"],
            has_fraud["observed_rate"],
            yerr=[
                has_fraud["observed_rate"] - has_fraud["observed_lower"].clip(lower=axis_min),
                has_fraud["observed_upper"] - has_fraud["observed_rate"],
            ],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.5,
            capsize=3,
            label=label,
        )
        if len(no_fraud):
            ax.errorbar(
                no_fraud["mean_predicted"],
                np.full(len(no_fraud), axis_min),
                yerr=[np.zeros(len(no_fraud)), no_fraud["observed_upper"] - axis_min],
                color=color,
                marker=marker,
                markersize=6,
                markerfacecolor=_SURFACE_COLOR,
                linestyle="none",
                capsize=3,
                clip_on=False,
            )

        edges = np.append(table["bin_lower"].to_numpy(), table["bin_upper"].iloc[-1])
        ax_hist.stairs(
            table["n"].to_numpy(),
            np.clip(edges, axis_min, 1),
            baseline=None,
            color=color,
            linewidth=1.5,
        )

    ax.set_yscale("log")
    ax.set_xlim(axis_min, 1)
    ax.set_ylim(axis_min, 1)
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.02,
        "hollow markers: bins with 0 frauds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=_NEUTRAL_COLOR,
    )
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Mean predicted probability")
    ax_hist.set_ylabel("Transactions")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
