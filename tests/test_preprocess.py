"""Tests for src/data/preprocess.py."""

import polars as pl
import pytest

from src.data.preprocess import (
    clean,
    time_aware_split,
    time_aware_train_val_test_split,
)


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


def test_three_way_split_is_time_ordered_across_slices() -> None:
    # Shuffled input, so the function has to do the sorting itself.
    times = [7, 3, 9, 0, 5, 1, 8, 2, 6, 4]
    lf = pl.LazyFrame({"Time": times})
    train_df, val_df, test_df = pl.collect_all(time_aware_train_val_test_split(lf))

    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()


def test_three_way_split_has_no_overlap_and_covers_every_row() -> None:
    lf = pl.LazyFrame({"Time": range(100), "id": range(100)})
    slices = pl.collect_all(time_aware_train_val_test_split(lf))
    ids = [set(df["id"].to_list()) for df in slices]

    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert ids[0] | ids[1] | ids[2] == set(range(100))


def test_three_way_split_respects_sizes() -> None:
    lf = pl.LazyFrame({"Time": range(100)})
    slices = pl.collect_all(
        time_aware_train_val_test_split(lf, val_size=0.1, test_size=0.3)
    )

    assert [df.height for df in slices] == [60, 10, 30]


def test_three_way_split_rounds_boundaries_down() -> None:
    # n = 7: floor(7 * 0.6) = 4 and floor(7 * 0.8) = 5 -> 4 / 1 / 2.
    lf = pl.LazyFrame({"Time": range(7)})
    slices = pl.collect_all(time_aware_train_val_test_split(lf, 0.2, 0.2))

    assert [df.height for df in slices] == [4, 1, 2]


@pytest.mark.parametrize("n", [7, 99, 100, 1_001, 2_837])
def test_three_way_test_slice_matches_two_way_test_slice(n: int) -> None:
    # The property that makes old (80/20) and new (60/20/20) test numbers
    # directly comparable: the test slice is unchanged, row for row.
    lf = pl.LazyFrame({"Time": [i // 3 for i in range(n)], "id": range(n)})
    _, old_test = pl.collect_all(time_aware_split(lf, test_size=0.2))
    _, _, new_test = pl.collect_all(
        time_aware_train_val_test_split(lf, val_size=0.2, test_size=0.2)
    )

    assert new_test.equals(old_test)


def test_three_way_split_keeps_tied_timestamps_in_original_order() -> None:
    lf = pl.LazyFrame({"Time": [2, 1, 1, 1, 2], "row": [0, 1, 2, 3, 4]})
    train_df, val_df, test_df = pl.collect_all(
        time_aware_train_val_test_split(lf, val_size=0.2, test_size=0.4)
    )

    assert train_df["row"].to_list() == [1, 2]
    assert val_df["row"].to_list() == [3]
    assert test_df["row"].to_list() == [0, 4]


@pytest.mark.parametrize(
    ("val_size", "test_size"),
    [(0.0, 0.2), (0.2, 0.0), (-0.1, 0.2), (0.5, 0.5), (0.6, 0.5)],
)
def test_three_way_split_rejects_invalid_sizes(
    val_size: float, test_size: float
) -> None:
    lf = pl.LazyFrame({"Time": range(10)})

    with pytest.raises(ValueError):
        time_aware_train_val_test_split(lf, val_size=val_size, test_size=test_size)
