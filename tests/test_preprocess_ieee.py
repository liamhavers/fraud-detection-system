"""Tests for src/data/preprocess_ieee.py."""

import polars as pl

from src.data.preprocess_ieee import (
    UNSEEN_CATEGORY_CODE,
    engineer_features,
    fit_feature_encoders,
    join_transaction_identity,
    time_aware_split,
)


def _make_transaction_lf() -> pl.LazyFrame:
    return pl.LazyFrame(
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
            "R_emaildomain": pl.Series([None] * 6, dtype=pl.String),
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


def _make_identity_lf() -> pl.LazyFrame:
    return pl.LazyFrame({"TransactionID": [1, 3, 5], "id_01": [0.1, 0.2, 0.3]})


def test_join_marks_has_identity_correctly() -> None:
    joined = join_transaction_identity(_make_transaction_lf(), _make_identity_lf()).collect()

    assert joined.height == 6
    assert dict(joined.select("TransactionID", "has_identity").iter_rows()) == {
        1: 1,
        2: 0,
        3: 1,
        4: 0,
        5: 1,
        6: 0,
    }


def test_time_aware_split_preserves_order() -> None:
    train_df, test_df = pl.collect_all(time_aware_split(_make_transaction_lf(), test_size=1 / 3))

    assert train_df["TransactionDT"].max() <= test_df["TransactionDT"].min()
    assert train_df.height + test_df.height == 6


def test_engineer_features_fits_only_on_train() -> None:
    joined = join_transaction_identity(_make_transaction_lf(), _make_identity_lf())
    train_lf, test_lf = time_aware_split(joined, test_size=1 / 3)
    train_fe, test_fe = pl.collect_all(engineer_features(train_lf, test_lf))

    # card 333 only appears in the test split -> unseen at fit time
    assert test_fe.filter(pl.col("card1") == 333)["card1_frequency"].to_list() == [0]
    assert train_fe["ProductCD"].dtype == pl.Int64
    assert test_fe["ProductCD"].dtype == pl.Int64
    # "R" only appears in test; R_emaildomain is entirely null in train.
    assert test_fe.filter(pl.col("card1") == 333)["ProductCD"].item() == UNSEEN_CATEGORY_CODE
    assert set(train_fe["R_emaildomain"]) == {UNSEEN_CATEGORY_CODE}


def test_time_since_last_txn_is_per_card_and_minus_one_for_first() -> None:
    train_lf, test_lf = time_aware_split(_make_transaction_lf(), test_size=1 / 3)
    train_fe, _ = pl.collect_all(engineer_features(train_lf, test_lf))

    # card 111 at DT 10, 20, 40; card 222 at DT 30
    assert dict(train_fe.select("TransactionDT", "time_since_last_txn_same_card").iter_rows()) == {
        10: -1.0,
        20: 10.0,
        30: -1.0,
        40: 20.0,
    }


def test_missing_indicator_ties_are_broken_by_column_order() -> None:
    lf = pl.LazyFrame(
        {
            "TransactionDT": [1, 2, 3, 4],
            "TransactionAmt": [1.0, 2.0, 3.0, 4.0],
            "card1": [1, 1, 2, 2],
            "z_mostly_null": [None, None, None, 1.0],
            "b_half_null": [None, None, 1.0, 1.0],
            "a_half_null": [None, None, 1.0, 1.0],
        }
    )
    encoders = fit_feature_encoders(lf)

    assert encoders["missing_indicator_columns"][:3] == [
        "z_mostly_null",
        "b_half_null",
        "a_half_null",
    ]
