"""Feature-distribution drift check between a reference window and new data.

Uses Population Stability Index (PSI) per feature; values above
DRIFT_PSI_THRESHOLD flag meaningful distribution shift worth investigating.

PSI is built from polars expressions rather than a per-column numpy loop:
one query computes every feature's quantile bin edges on the reference
window, one query per window computes every feature's bin shares, and the
PSI sum itself is an expression aggregated per feature. Polars runs each
query's expressions in parallel across features.
"""

from itertools import pairwise

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from src.config import DRIFT_PSI_THRESHOLD

# Bins with zero mass in either window would blow up the PSI log-ratio term;
# clamping to a small floor keeps a single empty bin from dominating the
# score while still penalising it.
_EPSILON = 1e-4


def _floor_empty_bin(share: pl.Expr) -> pl.Expr:
    return pl.when(share == 0).then(_EPSILON).otherwise(share)


def _psi(reference_share: pl.Expr, current_share: pl.Expr) -> pl.Expr:
    """PSI summed over bins: sum((cur - ref) * ln(cur / ref))."""
    ref, cur = _floor_empty_bin(reference_share), _floor_empty_bin(current_share)
    return ((cur - ref) * (cur / ref).log()).sum()


def _quantile_bin_edges(
    reference_lf: pl.LazyFrame, features: list[str], bins: int
) -> dict[str, list[float]]:
    """Per-feature bin edges at evenly spaced quantiles of the reference window.

    Duplicate edges (from repeated values) are collapsed, and the outer
    edges are opened to +/-inf so current-window values outside the
    reference range still land in the first/last bin.
    """
    quantiles = np.linspace(0, 1, bins + 1)
    edges = reference_lf.select(
        pl.concat_list(pl.col(f).quantile(q, interpolation="linear") for q in quantiles)
        .list.unique()
        .list.sort()
        .alias(f)
        for f in features
    ).collect()
    return {f: [-np.inf, *edges[f][0].to_list()[1:-1], np.inf] for f in features}


def _bin_shares(lf: pl.LazyFrame, bin_edges: dict[str, list[float]]) -> pl.LazyFrame:
    """Share of each feature's non-null values in each bin, as (feature, share) rows.

    Bins are half-open [lo, hi) except the last, which is closed —
    the same convention as np.histogram.
    """

    def share(feature: str, lo: float, hi: float, is_last: bool) -> pl.Expr:
        closed = "both" if is_last else "left"
        return pl.col(feature).drop_nulls().is_between(lo, hi, closed=closed).mean()

    return lf.select(
        pl.concat_list(
            share(f, lo, hi, is_last=(i == len(edges) - 2))
            for i, (lo, hi) in enumerate(pairwise(edges))
        ).alias(f)
        for f, edges in bin_edges.items()
    ).unpivot(variable_name="feature", value_name="share")


def _psi_by_feature(
    reference_lf: pl.LazyFrame, current_lf: pl.LazyFrame, features: list[str], bins: int
) -> pl.DataFrame:
    """PSI for each feature, as a (feature, psi) frame in `features` order."""
    bin_edges = _quantile_bin_edges(reference_lf, features, bins)
    # Fewer than two bins means a (near-)constant reference: there's no
    # distribution to compare against, so report zero drift.
    binnable = {f: edges for f, edges in bin_edges.items() if len(edges) >= 3}

    psi = (
        _bin_shares(reference_lf, binnable)
        .join(_bin_shares(current_lf, binnable), on="feature", suffix="_current")
        .explode("share", "share_current", empty_as_null=False)
        .group_by("feature")
        .agg(_psi(pl.col("share"), pl.col("share_current")).alias("psi"))
    )
    return (
        pl.LazyFrame({"feature": features})
        .join(psi, on="feature", how="left", maintain_order="left")
        .with_columns(pl.col("psi").fill_null(0.0))
        .collect()
    )


def population_stability_index(
    reference: ArrayLike, current: ArrayLike, bins: int = 10
) -> float:
    """Compute PSI between a reference and current feature distribution.

    Bin edges are quantiles of the reference distribution, so each reference
    bin holds roughly equal mass by construction; PSI then measures how far
    the current distribution's mass has shifted across those same bins.
    """
    reference_lf = pl.LazyFrame({"value": np.asarray(reference, dtype=float)})
    current_lf = pl.LazyFrame({"value": np.asarray(current, dtype=float)})
    return _psi_by_feature(reference_lf, current_lf, ["value"], bins)["psi"].item()


def check_drift(
    reference_lf: pl.LazyFrame, current_lf: pl.LazyFrame, threshold: float = DRIFT_PSI_THRESHOLD
) -> pl.DataFrame:
    """Return a per-feature PSI report flagging features that have drifted."""
    reference_schema, current_schema = reference_lf.collect_schema(), current_lf.collect_schema()
    shared_numeric_columns = [
        col
        for col, dtype in reference_schema.items()
        if col in current_schema and dtype.is_numeric() and current_schema[col].is_numeric()
    ]

    return (
        _psi_by_feature(reference_lf, current_lf, shared_numeric_columns, bins=10)
        .with_columns((pl.col("psi") > threshold).alias("drifted"))
        .sort("psi", descending=True, maintain_order=True)
    )


if __name__ == "__main__":
    from src.data.load import scan_raw_data
    from src.data.preprocess import clean, time_aware_split

    reference_lf, current_lf = time_aware_split(clean(scan_raw_data()))
    reference_df, current_df = (
        df.drop("Class") for df in pl.collect_all([reference_lf, current_lf])
    )

    with pl.Config(tbl_rows=-1):
        print("Reference (train) vs. current (test) — the real time-boundary check:")
        print(check_drift(reference_df.lazy(), current_df.lazy()))

        # A synthetic "new" window with injected drift, to demonstrate the check
        # actually catches a real shift rather than always reporting near-zero
        # PSI on this dataset's naturally similar train/test windows.
        drifted_lf = current_df.lazy().with_columns(pl.col("Amount") * 3 + 50)

        print("\nReference (train) vs. a simulated drifted window (Amount inflated 3x + 50):")
        print(check_drift(reference_df.lazy(), drifted_lf))
