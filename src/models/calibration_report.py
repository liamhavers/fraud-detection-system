"""Phase 7 calibration report: `python -m src.models.calibration_report`.

This is the one place the Kaggle test slice is used, for final reporting.
Everything that is *chosen* (calibrator parameters, Platt vs isotonic,
operating thresholds) is chosen on the validation slice first.

Steps:
1. Fit Platt and isotonic calibrators for each XGBoost variant on validation,
   and pick each model's method by cross-validated log loss on validation.
2. Score all 9 (model × raw/Platt/isotonic) combinations on test: ranking
   (PR-AUC) and level (Brier, log loss, O/E, slope/intercept, ECE).
3. Compare operating thresholds by test cost: (a) searched on validation,
   (b) analytic fp/(fp + fn), (c) an oracle searched on test, which
   reproduces the old optimistic approach and is reference only.
4. Compare fitted Platt (a, b) against the prior-shift prediction.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    CALIBRATION_BIN_EDGES,
    MODEL_ARTIFACT_CLASS_WEIGHTED_FILE,
    MODEL_ARTIFACT_SMOTE_FILE,
    MODEL_ARTIFACT_UNWEIGHTED_FILE,
    REPORTS_DIR,
)
from src.data.load import scan_raw_data
from src.data.preprocess import clean, time_aware_train_val_test_split
from src.models.calibrate import (
    Calibrator,
    analytic_threshold,
    brier_score,
    brier_skill_score,
    calibration_in_the_large,
    calibration_slope_intercept,
    calibrator_artifact_path,
    cross_validated_log_loss,
    expected_calibration_error,
    log_loss_score,
    reliability_table,
)
from src.models.evaluate import (
    confusion_counts,
    expected_cost,
    fine_threshold_grid,
    plot_reliability_diagram,
    pr_auc_score,
    select_cost_minimising_threshold,
)
from src.models.train import log_experiment

MODEL_FILES: dict[str, Path] = {
    "unweighted": MODEL_ARTIFACT_UNWEIGHTED_FILE,
    "class_weighted": MODEL_ARTIFACT_CLASS_WEIGHTED_FILE,
    "smote": MODEL_ARTIFACT_SMOTE_FILE,
}
CALIBRATION_METHODS = ("platt", "isotonic")
RAW_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)  # the original Phase 3 grid


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Ranking and calibration metrics for one set of predictions."""
    table = reliability_table(y, p, CALIBRATION_BIN_EDGES)
    return {
        "pr_auc": pr_auc_score(y, p),
        "brier": brier_score(y, p),
        "brier_skill": brier_skill_score(y, p),
        "log_loss": log_loss_score(y, p),
        **calibration_in_the_large(y, p),
        **calibration_slope_intercept(y, p),
        "ece": expected_calibration_error(table),
    }


def threshold_comparison(
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    grid: np.ndarray,
) -> dict[str, dict]:
    """Test cost at the validation-searched, analytic and oracle thresholds."""
    naive_cost = expected_cost(y_test, np.zeros(len(y_test), dtype=int))
    thresholds = {
        "validation": select_cost_minimising_threshold(y_val, p_val, thresholds=grid),
        "analytic": analytic_threshold(),
        "oracle_test": select_cost_minimising_threshold(y_test, p_test, thresholds=grid),
    }
    results = {}
    for name, t in thresholds.items():
        y_pred = (p_test >= t).astype(int)
        cost = expected_cost(y_test, y_pred)
        results[name] = {
            "threshold": t,
            "test_cost": cost,
            "cost_reduction_vs_naive": (naive_cost - cost) / naive_cost,
            **confusion_counts(y_test, y_pred),
        }
    return results


def prior_shift_prediction(model_name: str, train_rate: float) -> float:
    """Predicted Platt intercept b if the only distortion is the training prior.

    Training on a fraud rate π' instead of π shifts every log-odds score by
    logit(π') - logit(π), so Platt should learn a = 1 and b = -(that shift).
    - class-weighted: positives weighted by w = n_neg / n_pos, so π' is 50%
      in effect and b = -log(w).
    - SMOTE: data resampled to 50/50, so b = logit(π) = log(π / (1 - π)).
      Since w = (1 - π) / π, these two predictions are the same number.
    - unweighted: trained on the true rate, so b = 0.
    """
    if model_name == "unweighted":
        return 0.0
    return float(np.log(train_rate / (1 - train_rate)))


def _markdown_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """Render rows as a markdown table. columns: (key, header, format)."""
    header = "| " + " | ".join(h for _, h, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(format(row[key], fmt) for key, _, fmt in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def main() -> dict:
    print("Loading the three-way split (test is used once, below, for reporting)...")
    train_df, val_df, test_df = (
        df.to_pandas()
        for df in pl.collect_all(time_aware_train_val_test_split(clean(scan_raw_data())))
    )
    X_val, y_val = val_df.drop(columns=["Class"]), val_df["Class"].to_numpy()
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"].to_numpy()
    train_rate = float(train_df["Class"].mean())
    print(
        f"  frauds: train {int(train_df['Class'].sum())}, "
        f"val {int(y_val.sum())}, test {int(y_test.sum())}"
    )

    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    platt_rows: list[dict] = []
    chosen_methods: dict[str, str] = {}
    cv_log_loss: dict[str, dict[str, float]] = {}
    raw_tables: dict[str, pd.DataFrame] = {}
    calibrated_tables: dict[str, pd.DataFrame] = {}

    for model_name, path in MODEL_FILES.items():
        model = joblib.load(path)

        # --- Choices, all made on validation -------------------------------
        calibrators = {
            method: Calibrator.fit(method, model, X_val, y_val) for method in CALIBRATION_METHODS
        }
        for method, calibrator in calibrators.items():
            joblib.dump(calibrator, calibrator_artifact_path(model_name, method))
        cv_log_loss[model_name] = {
            method: cross_validated_log_loss(method, model, X_val, y_val)
            for method in CALIBRATION_METHODS
        }
        chosen = min(cv_log_loss[model_name], key=cv_log_loss[model_name].get)
        chosen_methods[model_name] = chosen

        platt = calibrators["platt"]
        platt_rows.append(
            {
                "model": model_name,
                "a": platt.platt_a,
                "b": platt.platt_b,
                "b_prior_shift": prior_shift_prediction(model_name, train_rate),
            }
        )

        # --- Reporting on test ---------------------------------------------
        val_raw = model.predict_proba(X_val)[:, 1]
        test_raw = model.predict_proba(X_test)[:, 1]
        variants = {"raw": (val_raw, test_raw, RAW_THRESHOLD_GRID)}
        for method, calibrator in calibrators.items():
            variants[method] = (
                calibrator.predict_proba(model, X_val),
                calibrator.predict_proba(model, X_test),
                fine_threshold_grid(),
            )

        for variant, (p_val, p_test, grid) in variants.items():
            label = {"model": model_name, "calibration": variant}
            metric_rows.append({**label, **probability_metrics(y_test, p_test)})
            for rule, result in threshold_comparison(y_val, p_val, y_test, p_test, grid).items():
                threshold_rows.append({**label, "rule": rule, **result})

        raw_tables[model_name] = reliability_table(y_test, test_raw, CALIBRATION_BIN_EDGES)
        calibrated_tables[f"{model_name} · {chosen}"] = reliability_table(
            y_test, variants[chosen][1], CALIBRATION_BIN_EDGES
        )

    plot_reliability_diagram(
        raw_tables,
        REPORTS_DIR / "reliability_raw.png",
        title="Reliability on test — raw scores",
    )
    plot_reliability_diagram(
        calibrated_tables,
        REPORTS_DIR / "reliability_calibrated.png",
        title="Reliability on test — calibrated on validation",
    )

    report = {
        "metrics_on_test": metric_rows,
        "thresholds": threshold_rows,
        "platt_parameters": platt_rows,
        "cv_log_loss_on_validation": cv_log_loss,
        "chosen_calibration_method": chosen_methods,
        "reliability_tables_on_test": {
            label: table.to_dict(orient="records")
            for label, table in {**raw_tables, **calibrated_tables}.items()
        },
        "n_fraud": {
            "train": int(train_df["Class"].sum()),
            "val": int(y_val.sum()),
            "test": int(y_test.sum()),
        },
    }
    log_experiment(
        "calibration_report",
        params={
            "bin_edges": list(CALIBRATION_BIN_EDGES),
            "analytic_threshold": analytic_threshold(),
            "train_fraud_rate": train_rate,
        },
        metrics=report,
    )

    print("\n## Probability metrics on test\n")
    print(
        _markdown_table(
            metric_rows,
            [
                ("model", "Model", ""),
                ("calibration", "Calibration", ""),
                ("pr_auc", "PR-AUC", ".3f"),
                ("brier", "Brier", ".2e"),
                ("brier_skill", "Brier skill", ".3f"),
                ("log_loss", "Log loss", ".5f"),
                ("o_to_e", "O/E", ".2f"),
                ("slope", "Slope", ".2f"),
                ("intercept", "Intercept", ".2f"),
                ("ece", "ECE", ".2e"),
            ],
        )
    )
    print("\n## Thresholds: expected cost on test (never-flag baseline: "
          f"{expected_cost(y_test, np.zeros(len(y_test), dtype=int)):,.0f})\n")
    print(
        _markdown_table(
            threshold_rows,
            [
                ("model", "Model", ""),
                ("calibration", "Calibration", ""),
                ("rule", "Threshold rule", ""),
                ("threshold", "Threshold", ".4f"),
                ("test_cost", "Test cost", ",.0f"),
                ("cost_reduction_vs_naive", "Reduction vs naive", ".1%"),
                ("true_positive", "TP", "d"),
                ("false_positive", "FP", "d"),
                ("false_negative", "FN", "d"),
            ],
        )
    )
    print("\n## Platt parameters (fitted on validation) vs prior-shift prediction\n")
    print(
        _markdown_table(
            platt_rows,
            [
                ("model", "Model", ""),
                ("a", "a (fitted)", ".3f"),
                ("b", "b (fitted)", ".2f"),
                ("b_prior_shift", "b (prior-shift prediction)", ".2f"),
            ],
        )
    )
    print("\n## Calibration method chosen by 5-fold CV log loss on validation\n")
    for model_name, losses in cv_log_loss.items():
        print(
            f"- {model_name}: platt {losses['platt']:.5f}, isotonic {losses['isotonic']:.5f}"
            f" -> {chosen_methods[model_name]}"
        )
    return report


if __name__ == "__main__":
    main()
