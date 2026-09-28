"""Cleaning, feature engineering, and time-aware train/test split."""

import pandas as pd

from src.config import TEST_SIZE


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and handle any nulls in the raw dataset."""
    return df.drop_duplicates().dropna().reset_index(drop=True)


def time_aware_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by the `Time` column so training data precedes test data.

    Deliberately not a random shuffle split: this mirrors a real production
    scenario where a model is trained on past transactions and evaluated on
    future ones.
    """
    df_sorted = df.sort_values("Time")
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    return train_df, test_df
