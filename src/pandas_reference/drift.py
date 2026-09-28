"""Feature-distribution drift check between a reference window and new data.

Uses Population Stability Index (PSI) per feature; values above
DRIFT_PSI_THRESHOLD flag meaningful distribution shift worth investigating.
"""

import numpy as np
import pandas as pd

from src.config import DRIFT_PSI_THRESHOLD

# Bins with zero mass in either window would blow up the PSI log-ratio term;
# clamping to a small floor keeps a single empty bin from dominating the
# score while still penalising it.
_EPSILON = 1e-4


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Compute PSI between a reference and current feature distribution.

    Bin edges are quantiles of the reference distribution, so each reference
    bin holds roughly equal mass by construction; PSI then measures how far
    the current distribution's mass has shifted across those same bins.
    """
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)

    bin_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(bin_edges) < 3:
        # Reference is (near-)constant — no meaningful distribution to bin.
        return 0.0
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=bin_edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=bin_edges)[0] / len(current)

    ref_pct = np.where(ref_pct == 0, _EPSILON, ref_pct)
    cur_pct = np.where(cur_pct == 0, _EPSILON, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = DRIFT_PSI_THRESHOLD
) -> pd.DataFrame:
    """Return a per-feature PSI report flagging features that have drifted."""
    shared_numeric_columns = [
        col
        for col in reference_df.columns
        if col in current_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[col])
        and pd.api.types.is_numeric_dtype(current_df[col])
    ]

    report = pd.DataFrame(
        [
            {
                "feature": col,
                "psi": population_stability_index(
                    reference_df[col].dropna().to_numpy(),
                    current_df[col].dropna().to_numpy(),
                ),
            }
            for col in shared_numeric_columns
        ]
    )
    report["drifted"] = report["psi"] > threshold
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.pandas_reference.load import load_raw_data
    from src.pandas_reference.preprocess import clean, time_aware_split

    df = clean(load_raw_data())
    reference_df, current_df = time_aware_split(df)

    print("Reference (train) vs. current (test) — the real time-boundary check:")
    print(check_drift(reference_df.drop(columns=["Class"]), current_df.drop(columns=["Class"])))

    # A synthetic "new" window with injected drift, to demonstrate the check
    # actually catches a real shift rather than always reporting near-zero
    # PSI on this dataset's naturally similar train/test windows.
    drifted_df = current_df.drop(columns=["Class"]).copy()
    drifted_df["Amount"] = drifted_df["Amount"] * 3 + 50

    print("\nReference (train) vs. a simulated drifted window (Amount inflated 3x + 50):")
    print(check_drift(reference_df.drop(columns=["Class"]), drifted_df))
