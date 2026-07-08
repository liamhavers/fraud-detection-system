"""Tests for src/data/preprocess.py."""

import pandas as pd

from src.data.preprocess import time_aware_split


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
