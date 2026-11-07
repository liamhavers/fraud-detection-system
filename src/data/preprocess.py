"""Cleaning and time-aware train/test split, as polars LazyFrame transforms.

Every function takes and returns a LazyFrame, so the whole path from CSV
scan to split is one query plan that polars optimises and runs in parallel
only when the caller collects it. Collect both halves of a split together
(`pl.collect_all([train_lf, test_lf])`) so the shared scan/clean/sort work
runs once rather than once per split.
"""

import polars as pl

from src.config import TEST_SIZE, VAL_SIZE


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


def time_aware_train_val_test_split(
    lf: pl.LazyFrame, val_size: float = VAL_SIZE, test_size: float = TEST_SIZE
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Split by `Time` into train, then validation, then test.

    Same approach as `time_aware_split` (stable sort, row index, boundaries
    rounded down), and the test boundary uses the same expression, so the
    test slice is row-for-row identical to `time_aware_split`'s. Validation
    is used for model selection, threshold search and calibration; test is
    held back for final reporting only.
    """
    if not (0 < val_size and 0 < test_size and val_size + test_size < 1):
        raise ValueError(
            "need 0 < val_size, 0 < test_size and val_size + test_size < 1, "
            f"got val_size={val_size}, test_size={test_size}"
        )
    ordered = lf.sort("Time", maintain_order=True).with_row_index("_row")
    # Round the fraction first: in floating point 1 - 0.4 - 0.2 is
    # 0.39999999999999997, which would floor 5 * 0.4 down to 1 instead of 2.
    train_fraction = round(1 - test_size - val_size, 12)
    val_idx = (pl.len() * train_fraction).floor()
    test_idx = (pl.len() * (1 - test_size)).floor()
    row = pl.col("_row")
    train_lf = ordered.filter(row < val_idx).drop("_row")
    val_lf = ordered.filter((row >= val_idx) & (row < test_idx)).drop("_row")
    test_lf = ordered.filter(row >= test_idx).drop("_row")
    return train_lf, val_lf, test_lf
