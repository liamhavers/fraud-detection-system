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


def paired_bootstrap_cost_difference(
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    n_bootstrap: int = AB_TEST_BOOTSTRAP_SAMPLES,
    confidence_level: float = AB_TEST_CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Paired bootstrap of the mean cost difference (A - B).

    In shadow mode both policies score the *same* transactions, so the data
    is paired: resample transactions (one index draw), and take the
    per-transaction difference d_i = cost_a_i - cost_b_i. Transactions where
    the policies agree have d_i = 0 and add no noise, so only disagreements
    drive the interval. The unpaired `bootstrap_cost_difference` resamples
    the two arrays independently, which ignores how strongly the two costs
    are correlated (both policies miss the same hard frauds) and gives a far
    wider interval. This is the bootstrap analogue of a paired t-test vs a
    two-sample t-test.
    """
    cost_a, cost_b = np.asarray(cost_a, dtype=float), np.asarray(cost_b, dtype=float)
    if cost_a.shape != cost_b.shape:
        raise ValueError("paired comparison needs one cost per transaction for each policy")
    diff = cost_a - cost_b
    rng = np.random.default_rng(random_state)

    batch_size = min(n_bootstrap, 200)  # see bootstrap_cost_difference on memory
    diffs = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        idx = rng.integers(0, len(diff), size=(end - start, len(diff)))
        diffs[start:end] = diff[idx].mean(axis=1)

    alpha = 1 - confidence_level
    ci_lower, ci_upper = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": float(diff.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_discordant": int(np.count_nonzero(diff)),
    }


def compare_policies(
    y_true: pd.Series,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    label_a: str = "smote",
    label_b: str = "class_weighted",
) -> dict:
    """Full shadow-mode comparison: expected cost per policy, paired
    bootstrapped CI on the difference, and whether it's statistically
    distinguishable from noise.

    The unpaired CI is returned alongside for reference only, to show how
    much the paired design narrows the interval (Phase 3 originally used it).
    """
    cost_a = per_transaction_cost(y_true, y_pred_a)
    cost_b = per_transaction_cost(y_true, y_pred_b)
    bootstrap_result = paired_bootstrap_cost_difference(cost_a, cost_b)
    unpaired = bootstrap_cost_difference(cost_a, cost_b)

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
        "unpaired_ci_lower": float(unpaired["ci_lower"]),
        "unpaired_ci_upper": float(unpaired["ci_upper"]),
    }




def _print_comparison(title: str, comparison: dict) -> None:
    a, b = comparison["label_a"], comparison["label_b"]
    print(f"\n[shadow-mode A/B test: {title}]")
    print(f"  mean cost/txn — {a}: {comparison['mean_cost_a']:.4f}, {b}: {comparison['mean_cost_b']:.4f}")
    print(
        f"  point estimate ({a} - {b}): {comparison['point_estimate']:.4f} "
        f"({comparison['relative_effect_size']:.1%} relative effect size)"
    )
    print(f"  transactions where the policies disagree: {comparison['n_discordant']}")
    print(
        f"  {comparison['confidence_level']:.0%} paired bootstrap CI: "
        f"[{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}]"
    )
    print(
        f"  (unpaired CI, reference only: "
        f"[{comparison['unpaired_ci_lower']:.4f}, {comparison['unpaired_ci_upper']:.4f}])"
    )
    print(f"  statistically significant: {comparison['statistically_significant']}")


if __name__ == "__main__":
    import joblib
    import polars as pl

    from src.config import (
        CALIBRATION_METHOD,
        MODEL_ARTIFACT_CLASS_WEIGHTED_FILE,
        MODEL_ARTIFACT_SMOTE_FILE,
        MODEL_ARTIFACT_UNWEIGHTED_FILE,
        REPORTS_DIR,
    )
    from src.data.load import scan_raw_data
    from src.data.preprocess import clean, time_aware_train_val_test_split
    from src.models.calibrate import Calibrator, analytic_threshold
    from src.models.evaluate import (
        confusion_counts,
        expected_cost,
        plot_confusion_matrix,
        plot_precision_recall_curves,
        pr_auc_score,
        select_cost_minimising_threshold,
    )
    from src.models.train import log_experiment

    # Every choice (thresholds, calibrator, which calibrated policy) is made
    # on the validation slice; test is used only to evaluate those choices.
    print("Loading the validation and test slices and the trained policies...")
    _, val_df, test_df = (
        df.to_pandas()
        for df in pl.collect_all(time_aware_train_val_test_split(clean(scan_raw_data())))
    )
    X_val, y_val = val_df.drop(columns=["Class"]), val_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]
    naive_cost = expected_cost(y_test, np.zeros(len(y_test), dtype=int))

    models = {
        "unweighted": joblib.load(MODEL_ARTIFACT_UNWEIGHTED_FILE),
        "class_weighted": joblib.load(MODEL_ARTIFACT_CLASS_WEIGHTED_FILE),
        "smote": joblib.load(MODEL_ARTIFACT_SMOTE_FILE),
    }

    def summarise(
        label: str, proba_test: np.ndarray, threshold: float, source: str = "chosen on validation"
    ) -> dict:
        y_pred = (proba_test >= threshold).astype(int)
        cost = expected_cost(y_test, y_pred)
        result = {
            "pr_auc": pr_auc_score(y_test, proba_test),
            "threshold": threshold,
            "expected_cost": cost,
            "cost_reduction_vs_naive": (naive_cost - cost) / naive_cost,
            "confusion_matrix": confusion_counts(y_test, y_pred),
            "y_pred": y_pred,
        }
        print(f"\n[{label}]")
        print(f"  PR-AUC (test): {result['pr_auc']:.4f}")
        print(f"  threshold ({source}): {threshold:.4f}")
        print(f"  expected cost (test): {cost:,.0f} (naive 'never flag' baseline: {naive_cost:,.0f})")
        print(f"  cost reduction vs naive baseline: {result['cost_reduction_vs_naive']:.1%}")
        print(f"  confusion matrix: {result['confusion_matrix']}")
        return result

    # --- Comparison 1: raw SMOTE vs raw class-weighting (the Phase 3 test,
    # now with thresholds searched on validation instead of test). ---------
    results: dict[str, dict] = {}
    proba_by_policy: dict[str, np.ndarray] = {}
    for label in ("class_weighted", "smote"):
        model = models[label]
        threshold = select_cost_minimising_threshold(y_val, model.predict_proba(X_val)[:, 1])
        proba_by_policy[label] = model.predict_proba(X_test)[:, 1]
        results[label] = summarise(label, proba_by_policy[label], threshold)
        plot_confusion_matrix(
            y_test,
            results[label]["y_pred"],
            REPORTS_DIR / f"confusion_matrix_{label}.png",
            title=f"{label} @ threshold={threshold:.2f}",
        )
    plot_precision_recall_curves(y_test, proba_by_policy, REPORTS_DIR / "pr_curve_kaggle.png")

    raw_comparison = compare_policies(
        y_test,
        results["smote"]["y_pred"],
        results["class_weighted"]["y_pred"],
        label_a="smote",
        label_b="class_weighted",
    )
    _print_comparison("smote vs. class_weighted (raw scores)", raw_comparison)

    # --- Comparison 2: best calibrated policy at the analytic threshold vs
    # raw class-weighting at its validation threshold. "Best" = lowest
    # validation cost at the analytic threshold, so test plays no part. ----
    t_star = analytic_threshold()
    calibrators = {
        name: Calibrator.fit(CALIBRATION_METHOD, model, X_val, y_val)
        for name, model in models.items()
    }
    val_costs = {
        name: expected_cost(
            y_val, (calibrators[name].predict_proba(models[name], X_val) >= t_star).astype(int)
        )
        for name in models
    }
    best = min(val_costs, key=val_costs.get)
    print(f"\nValidation cost at the analytic threshold ({t_star:.4f}), {CALIBRATION_METHOD}-calibrated:")
    for name, cost in val_costs.items():
        print(f"  {name}: {cost:,.0f}" + ("  <- chosen" if name == best else ""))

    calibrated_label = f"{best}_{CALIBRATION_METHOD}"
    results[calibrated_label] = summarise(
        calibrated_label,
        calibrators[best].predict_proba(models[best], X_test),
        t_star,
        source="analytic, fp / (fp + fn)",
    )
    calibrated_comparison = compare_policies(
        y_test,
        results[calibrated_label]["y_pred"],
        results["class_weighted"]["y_pred"],
        label_a=calibrated_label,
        label_b="class_weighted",
    )
    _print_comparison(
        f"{calibrated_label} @ analytic threshold vs. class_weighted (raw)",
        calibrated_comparison,
    )

    log_experiment(
        "shadow_mode_ab_test",
        params={
            "fn_cost": COST_FALSE_NEGATIVE,
            "fp_cost": COST_FALSE_POSITIVE,
            "n_bootstrap": AB_TEST_BOOTSTRAP_SAMPLES,
            "confidence_level": AB_TEST_CONFIDENCE_LEVEL,
            "bootstrap": "paired",
            "threshold_selection": "validation",
            "calibration_method": CALIBRATION_METHOD,
        },
        metrics={
            **{
                label: {k: v for k, v in result.items() if k != "y_pred"}
                for label, result in results.items()
            },
            "validation_cost_at_analytic_threshold": val_costs,
            "comparison_raw": raw_comparison,
            "comparison_calibrated": calibrated_comparison,
            "naive_baseline_cost": naive_cost,
        },
    )
