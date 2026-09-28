"""Tests for src/data/preprocess.py."""

import polars as pl

from src.data.preprocess import clean, time_aware_split


def test_clean_drops_duplicates_and_nulls() -> None:
    lf = pl.LazyFrame(
        {
            "Time": [1, 1, 2, 3],
            "Amount": [10.0, 10.0, 20.0, None],
        }
    )
    cleaned = clean(lf).collect()

    assert cleaned.height == 2
    assert cleaned["Amount"].null_count() == 0


def test_clean_keeps_first_occurrence_in_original_order() -> None:
    lf = pl.LazyFrame({"Time": [3, 1, 3, 2], "Amount": [5.0, 1.0, 5.0, 2.0]})

    assert clean(lf).collect()["Time"].to_list() == [3, 1, 2]


def test_time_aware_split_preserves_order() -> None:
    lf = pl.LazyFrame({"Time": range(100), "Amount": range(100)})
    train_df, test_df = pl.collect_all(time_aware_split(lf, test_size=0.2))

    assert train_df["Time"].max() <= test_df["Time"].min()
    assert train_df.height + test_df.height == 100


def test_time_aware_split_respects_test_size() -> None:
    lf = pl.LazyFrame({"Time": range(100), "Amount": range(100)})
    train_df, test_df = pl.collect_all(time_aware_split(lf, test_size=0.3))

    assert test_df.height == 30
    assert train_df.height == 70


def test_time_aware_split_rounds_the_boundary_down() -> None:
    # 7 * 0.8 = 5.6 -> 5 training rows, matching int() truncation.
    lf = pl.LazyFrame({"Time": range(7)})
    train_df, test_df = pl.collect_all(time_aware_split(lf, test_size=0.2))

    assert (train_df.height, test_df.height) == (5, 2)


def test_time_aware_split_keeps_tied_timestamps_in_original_order() -> None:
    lf = pl.LazyFrame({"Time": [2, 1, 1, 1, 2], "row": [0, 1, 2, 3, 4]})
    train_df, test_df = pl.collect_all(time_aware_split(lf, test_size=0.4))

    assert train_df["row"].to_list() == [1, 2, 3]
    assert test_df["row"].to_list() == [0, 4]
