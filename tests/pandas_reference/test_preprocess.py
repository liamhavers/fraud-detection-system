"""Tests for src/pandas_reference/preprocess.py."""

import numpy as np
import pandas as pd

from src.pandas_reference.preprocess import clean, time_aware_split


def test_clean_drops_duplicates_and_nulls() -> None:
    df = pd.DataFrame(
        {
            "Time": [1, 1, 2, 3],
            "Amount": [10.0, 10.0, 20.0, np.nan],
        }
    )
    cleaned = clean(df)

    assert len(cleaned) == 2
    assert cleaned["Amount"].isnull().sum() == 0


def test_time_aware_split_preserves_order() -> None:
    df = pd.DataFrame({"Time": range(100), "Amount": range(100)})
    train_df, test_df = time_aware_split(df, test_size=0.2)

    assert train_df["Time"].max() <= test_df["Time"].min()
    assert len(train_df) + len(test_df) == len(df)


def test_time_aware_split_respects_test_size() -> None:
    df = pd.DataFrame({"Time": range(100), "Amount": range(100)})
    train_df, test_df = time_aware_split(df, test_size=0.3)

    assert len(test_df) == 30
    assert len(train_df) == 70
