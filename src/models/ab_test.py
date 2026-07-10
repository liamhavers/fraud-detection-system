"""Offline shadow-mode statistical comparison between two trained policies.

This replays historical held-out data through two competing policies
(SMOTE-resampled model vs. class-weighted model) and compares their expected
cost using bootstrapped confidence intervals, rather than picking a winner by
eyeballing metrics.

This is deliberately an offline simulation for building statistical
intuition — not a live, traffic-splitting production experiment. It has no
real traffic split and no protection against novelty or seasonality effects
a real online test would need to control for. See README "Why a shadow-mode
A/B test" for the full framing.
"""

import numpy as np
import pandas as pd

from src.config import (
    AB_TEST_BOOTSTRAP_SAMPLES,
    AB_TEST_CONFIDENCE_LEVEL,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    RANDOM_STATE,
)


def per_transaction_cost(
    y_true: pd.Series,
    y_pred: np.ndarray,
    fn_cost: float = COST_FALSE_NEGATIVE,
    fp_cost: float = COST_FALSE_POSITIVE,
) -> np.ndarray:
    """Cost incurred by each transaction under a given policy's predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cost = np.zeros(len(y_true))
    cost[(y_true == 1) & (y_pred == 0)] = fn_cost
    cost[(y_true == 0) & (y_pred == 1)] = fp_cost
    return cost


def bootstrap_cost_difference(
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    n_bootstrap: int = AB_TEST_BOOTSTRAP_SAMPLES,
    confidence_level: float = AB_TEST_CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Bootstrap the mean cost difference (A - B) and return point estimate + CI.

    A percentile bootstrap: resample each cost array with replacement
    n_bootstrap times, take the mean difference each time, then read off the
    confidence interval from the resulting distribution's percentiles.

    Run in batches rather than materialising all n_bootstrap resamples at
    once: a single (n_bootstrap, len(cost)) index array is cheap for the
    synthetic-data unit tests (len ~1,000) but explodes to tens of GB against
    a real ~57k-row test set at n_bootstrap=10,000 — that's what OOM-killed
    the process the first time this ran end-to-end.
    """
    rng = np.random.default_rng(random_state)
    cost_a, cost_b = np.asarray(cost_a), np.asarray(cost_b)

    batch_size = min(n_bootstrap, 200)
    diffs = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        idx_a = rng.integers(0, len(cost_a), size=(end - start, len(cost_a)))
        idx_b = rng.integers(0, len(cost_b), size=(end - start, len(cost_b)))
        diffs[start:end] = cost_a[idx_a].mean(axis=1) - cost_b[idx_b].mean(axis=1)

    alpha = 1 - confidence_level
    ci_lower, ci_upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": cost_a.mean() - cost_b.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
    }


def compare_policies(
    y_true: pd.Series,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    label_a: str = "smote",
    label_b: str = "class_weighted",
) -> dict:
    """Full shadow-mode comparison: expected cost per policy, bootstrapped CI on
    the difference, and whether it's statistically distinguishable from noise.
    """
    cost_a = per_transaction_cost(y_true, y_pred_a)
    cost_b = per_transaction_cost(y_true, y_pred_b)
    bootstrap_result = bootstrap_cost_difference(cost_a, cost_b)

    # CI excludes zero => the cost difference is distinguishable from noise
    # at this confidence level, not just a coin-flip in which policy is bigger.
    significant = bootstrap_result["ci_lower"] > 0 or bootstrap_result["ci_upper"] < 0
    relative_effect_size = bootstrap_result["point_estimate"] / max(cost_a.mean(), cost_b.mean())

    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_cost_a": float(cost_a.mean()),
        "mean_cost_b": float(cost_b.mean()),
        "relative_effect_size": float(relative_effect_size),
        "statistically_significant": bool(significant),
        **bootstrap_result,
    }


if __name__ == "__main__":
    import joblib

    from src.config import (
        MODEL_ARTIFACT_CLASS_WEIGHTED_FILE,
        MODEL_ARTIFACT_SMOTE_FILE,
        REPORTS_DIR,
    )
    from src.data.load import load_raw_data
    from src.data.preprocess import clean, time_aware_split
    from src.models.evaluate import (
        confusion_counts,
        expected_cost,
        plot_confusion_matrix,
        plot_precision_recall_curves,
        pr_auc_score,
        select_cost_minimising_threshold,
    )
    from src.models.train import log_experiment

    print("Loading held-out Kaggle test set and both trained policies...")
    df = clean(load_raw_data())
    _, test_df = time_aware_split(df)
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    policies = {
        "class_weighted": joblib.load(MODEL_ARTIFACT_CLASS_WEIGHTED_FILE),
        "smote": joblib.load(MODEL_ARTIFACT_SMOTE_FILE),
    }

    proba_by_policy: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    # Threshold is selected and evaluated on the same held-out test set here,
    # a deliberate simplification for this portfolio project's single
    # train/test split rather than carving out a third validation slice. In
    # production this would be tuned on a separate validation window to
    # avoid any risk of the operating point being tailored to this specific
    # test set's noise.
    naive_cost = expected_cost(y_test, np.zeros(len(y_test), dtype=int))

    for label, model in policies.items():
        proba = model.predict_proba(X_test)[:, 1]
        proba_by_policy[label] = proba
        threshold = select_cost_minimising_threshold(y_test, proba)
        y_pred = (proba >= threshold).astype(int)
        cost = expected_cost(y_test, y_pred)

        results[label] = {
            "pr_auc": pr_auc_score(y_test, proba),
            "threshold": threshold,
            "expected_cost": cost,
            "cost_reduction_vs_naive": (naive_cost - cost) / naive_cost,
            "confusion_matrix": confusion_counts(y_test, y_pred),
            "y_pred": y_pred,
        }

        print(f"\n[{label}]")
        print(f"  PR-AUC: {results[label]['pr_auc']:.4f}")
        print(f"  cost-minimising threshold: {threshold:.2f}")
        print(f"  expected cost: {cost:,.0f} (naive 'never flag' baseline: {naive_cost:,.0f})")
        print(f"  cost reduction vs naive baseline: {results[label]['cost_reduction_vs_naive']:.1%}")
        print(f"  confusion matrix: {results[label]['confusion_matrix']}")

        plot_confusion_matrix(
            y_test,
            y_pred,
            REPORTS_DIR / f"confusion_matrix_{label}.png",
            title=f"{label} @ threshold={threshold:.2f}",
        )

    plot_precision_recall_curves(y_test, proba_by_policy, REPORTS_DIR / "pr_curve_kaggle.png")

    comparison = compare_policies(
        y_test,
        results["smote"]["y_pred"],
        results["class_weighted"]["y_pred"],
        label_a="smote",
        label_b="class_weighted",
    )

    print("\n[shadow-mode A/B test: smote vs. class_weighted]")
    print(f"  mean cost/txn — smote: {comparison['mean_cost_a']:.4f}, class_weighted: {comparison['mean_cost_b']:.4f}")
    print(
        f"  point estimate (smote - class_weighted): {comparison['point_estimate']:.4f} "
        f"({comparison['relative_effect_size']:.1%} relative effect size)"
    )
    print(
        f"  {comparison['confidence_level']:.0%} bootstrap CI: "
        f"[{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}]"
    )
    print(f"  statistically significant: {comparison['statistically_significant']}")

    log_experiment(
        "shadow_mode_ab_test",
        params={
            "fn_cost": COST_FALSE_NEGATIVE,
            "fp_cost": COST_FALSE_POSITIVE,
            "n_bootstrap": AB_TEST_BOOTSTRAP_SAMPLES,
            "confidence_level": AB_TEST_CONFIDENCE_LEVEL,
        },
        metrics={
            "class_weighted": {k: v for k, v in results["class_weighted"].items() if k != "y_pred"},
            "smote": {k: v for k, v in results["smote"].items() if k != "y_pred"},
            "comparison": {k: v for k, v in comparison.items()},
            "naive_baseline_cost": naive_cost,
        },
    )
