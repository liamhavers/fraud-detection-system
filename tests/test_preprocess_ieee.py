"""Tests for src/data/preprocess_ieee.py."""

import pandas as pd

from src.data.preprocess_ieee import (
    engineer_features,
    join_transaction_identity,
    time_aware_split,
)


def _make_transaction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "TransactionAmt": [100.0, 50.0, 200.0, 75.0, 60.0, 90.0],
            "isFraud": [0, 0, 1, 0, 0, 1],
            "card1": [111, 111, 222, 111, 222, 333],
            "ProductCD": ["W", "C", "W", "W", "C", "R"],
            "card4": ["visa", "visa", "mastercard", "visa", "mastercard", None],
            "card6": ["debit", "debit", "credit", "debit", "credit", "credit"],
            "P_emaildomain": ["a.com", "b.com", "a.com", None, "b.com", "c.com"],
            "R_emaildomain": [None] * 6,
            "M1": ["T", "F", "T", "T", "F", None],
            "M2": ["T", "F", "T", "T", "F", None],
            "M3": ["T", "F", "T", "T", "F", None],
            "M4": ["M0", "M1", "M0", "M0", "M1", None],
            "M5": ["T", "F", "T", "T", "F", None],
            "M6": ["T", "F", "T", "T", "F", None],
            "M7": ["T", "F", "T", "T", "F", None],
            "M8": ["T", "F", "T", "T", "F", None],
            "M9": ["T", "F", "T", "T", "F", None],
        }
    )


def _make_identity_df() -> pd.DataFrame:
    return pd.DataFrame({"TransactionID": [1, 3, 5], "id_01": [0.1, 0.2, 0.3]})


def test_join_marks_has_identity_correctly() -> None:
    joined = join_transaction_identity(_make_transaction_df(), _make_identity_df())

    assert len(joined) == 6
    assert joined.set_index("TransactionID")["has_identity"].to_dict() == {
        1: 1,
        2: 0,
        3: 1,
        4: 0,
        5: 1,
        6: 0,
    }


def test_time_aware_split_preserves_order() -> None:
    df = _make_transaction_df()
    train_df, test_df = time_aware_split(df, test_size=1 / 3)

    assert train_df["TransactionDT"].max() <= test_df["TransactionDT"].min()
    assert len(train_df) + len(test_df) == len(df)


def test_engineer_features_fits_only_on_train() -> None:
    joined = join_transaction_identity(_make_transaction_df(), _make_identity_df())
    train_df, test_df = time_aware_split(joined, test_size=1 / 3)
    train_fe, test_fe = engineer_features(train_df, test_df)

    # card 333 only appears in the test split -> unseen at fit time
    assert (test_fe.loc[test_fe["card1"] == 333, "card1_frequency"] == 0).all()
    assert train_fe["ProductCD"].dtype == int
    assert test_fe["ProductCD"].dtype == int
