"""IEEE-CIS preprocessing: join, categorical encoding, missing-data strategy.

Kept separate from preprocess.py (Kaggle) rather than a shared generic
pipeline — the two datasets have genuinely different shapes (anonymised PCA
components vs. raw categorical/identity fields with heavy missingness) and a
forced-generic abstraction would obscure that rather than clarify it.
"""

import pandas as pd

from src.config import TEST_SIZE


def join_transaction_identity(
    transaction_df: pd.DataFrame, identity_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join transaction and identity tables on TransactionID."""
    raise NotImplementedError


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical encoding, missing-data flagging/imputation, aggregation features."""
    raise NotImplementedError


def time_aware_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by TransactionDT so training data precedes test data (see preprocess.py)."""
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    return train_df, test_df
