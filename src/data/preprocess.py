"""Cleaning and time-aware train/test split, as polars LazyFrame transforms.

Every function takes and returns a LazyFrame, so the whole path from CSV
scan to split is one query plan that polars optimises and runs in parallel
only when the caller collects it. Collect both halves of a split together
(`pl.collect_all([train_lf, test_lf])`) so the shared scan/clean/sort work
runs once rather than once per split.
"""

import polars as pl

from src.config import TEST_SIZE


def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop duplicate rows (keeping the first) and rows with any null."""
    return lf.unique(keep="first", maintain_order=True).drop_nulls()


def time_aware_split(
    lf: pl.LazyFrame, test_size: float = TEST_SIZE
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split by the `Time` column so training data precedes test data.

    Deliberately not a random shuffle split: this mirrors a real production
    scenario where a model is trained on past transactions and evaluated on
    future ones. The sort is stable, so rows sharing a timestamp keep their
    original order and the boundary row is deterministic.
    """
    ordered = lf.sort("Time", maintain_order=True).with_row_index("_row")
    split_idx = (pl.len() * (1 - test_size)).floor()
    train_lf = ordered.filter(pl.col("_row") < split_idx).drop("_row")
    test_lf = ordered.filter(pl.col("_row") >= split_idx).drop("_row")
    return train_lf, test_lf
