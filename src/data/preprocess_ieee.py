"""IEEE-CIS preprocessing: join, categorical encoding, missing-data strategy.

Kept separate from preprocess.py (Kaggle) rather than a shared generic
pipeline — the two datasets have genuinely different shapes (anonymised PCA
components vs. raw categorical/identity fields with heavy missingness) and a
forced-generic abstraction would obscure that rather than clarify it.

Feature engineering is deliberately fit on the training split only and then
applied to test (`fit_feature_encoders` / `transform_features`), rather than
fit on the whole joined dataset before splitting. Category vocabularies and
the card1 aggregation statistics below would otherwise leak test-period
information into training — the same class of leakage the Kaggle EDA
notebook found lurking in that dataset's near-duplicate rows.

Everything here is a polars LazyFrame transform. `fit_feature_encoders` is
the one place that has to execute a query, because its outputs (category
vocabularies, which columns get missingness flags) decide the shape of
the transformed frame. Everything else stays lazy until the caller collects.
Polars reads empty CSV fields as null rather than NaN, so null is the only
missing-value marker in this module.
"""

import polars as pl

from src.config import TEST_SIZE

CATEGORICAL_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]
UNSEEN_CATEGORY_CODE = -1
N_MISSING_INDICATOR_COLUMNS = 8


def join_transaction_identity(
    transaction_lf: pl.LazyFrame, identity_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """Left-join transaction and identity tables on TransactionID.

    Most transactions (~76%) have no matching identity row — that's a
    structural pattern worth keeping visible as its own feature rather than
    letting it collapse into "missing", since a transaction with no device/
    identity fingerprint at all is a different situation from one where a
    specific identity field happens to be null.
    """
    return transaction_lf.join(
        identity_lf.with_columns(pl.lit(1, dtype=pl.Int64).alias("has_identity")),
        on="TransactionID",
        how="left",
        maintain_order="left",
    ).with_columns(pl.col("has_identity").fill_null(0))


def time_aware_split(
    lf: pl.LazyFrame, test_size: float = TEST_SIZE
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split by TransactionDT so training data precedes test data (see preprocess.py)."""
    ordered = lf.sort("TransactionDT", maintain_order=True).with_row_index("_row")
    split_idx = (pl.len() * (1 - test_size)).floor()
    train_lf = ordered.filter(pl.col("_row") < split_idx).drop("_row")
    test_lf = ordered.filter(pl.col("_row") >= split_idx).drop("_row")
    return train_lf, test_lf


def fit_feature_encoders(train_lf: pl.LazyFrame) -> dict:
    """Fit categorical vocabularies and card1 aggregation stats on TRAIN ONLY.

    Returns a dict of fitted artifacts to be passed to transform_features()
    for both the train and test splits, so test never influences what the
    encoders learned. The three aggregation queries are collected together
    so polars scans the training split once for all of them.
    """
    train_columns = train_lf.collect_schema().names()
    categorical_columns_present = [col for col in CATEGORICAL_COLUMNS if col in train_columns]

    vocabularies_query = train_lf.select(
        pl.col(col).drop_nulls().unique().sort().implode() for col in categorical_columns_present
    )
    # Ties in null fraction are common here (identity fields go missing
    # together), so rank ties by column order to keep the selected
    # columns deterministic.
    null_fraction_query = (
        train_lf.select(pl.all().null_count() / pl.len())
        .unpivot(variable_name="column", value_name="null_fraction")
        .sort("null_fraction", descending=True, maintain_order=True)
        .head(N_MISSING_INDICATOR_COLUMNS)
    )
    card1_stats_query = (
        train_lf.filter(pl.col("card1").is_not_null())
        .group_by("card1")
        .agg(
            pl.len().cast(pl.Int64).alias("card1_frequency"),
            pl.col("TransactionAmt").mean().alias("card1_mean_amount"),
        )
        .sort("card1")
    )
    global_mean_query = train_lf.select(pl.col("TransactionAmt").mean())

    vocabularies, null_fraction, card1_stats, global_mean = pl.collect_all(
        [vocabularies_query, null_fraction_query, card1_stats_query, global_mean_query]
    )

    category_maps = {
        col: {category: code for code, category in enumerate(vocabularies[col][0])}
        for col in categorical_columns_present
    }

    return {
        "category_maps": category_maps,
        "card1_stats": card1_stats,
        "global_mean_amount": global_mean.item(),
        "missing_indicator_columns": null_fraction["column"].to_list(),
    }


def transform_features(lf: pl.LazyFrame, encoders: dict) -> pl.LazyFrame:
    """Apply encoders fitted by fit_feature_encoders() to a train or test split.

    Missing values in the underlying V/D/C feature blocks are left as null
    rather than imputed — XGBoost handles missing values natively by
    learning a default split direction per node, and imputing ~85%-null
    columns with a fabricated value would add noise, not signal. The
    indicator flags below exist for the columns where missingness itself
    looks informative enough to surface explicitly (see EDA notebook).
    """
    # Missingness flags must be computed before categorical encoding, which
    # replaces nulls with UNSEEN_CATEGORY_CODE — several of the most-null
    # columns (e.g. id_23, id_27) are categorical.
    missing_indicators = [
        pl.col(col).is_null().cast(pl.Int64).alias(f"{col}_is_missing")
        for col in encoders["missing_indicator_columns"]
    ]
    categorical_codes = [
        pl.col(col).replace_strict(mapping, default=UNSEEN_CATEGORY_CODE, return_dtype=pl.Int64)
        for col, mapping in encoders["category_maps"].items()
    ]
    time_since_last_txn = (
        pl.when(pl.col("card1").is_not_null())
        .then(pl.col("TransactionDT").diff().over("card1"))
        .cast(pl.Float64)
        .fill_null(-1)
        .alias("time_since_last_txn_same_card")
    )

    return (
        lf.with_columns(missing_indicators)
        .with_columns(categorical_codes)
        .join(encoders["card1_stats"].lazy(), on="card1", how="left", maintain_order="left")
        .with_columns(
            pl.col("card1_frequency").fill_null(0),
            pl.col("card1_mean_amount").fill_null(encoders["global_mean_amount"]),
        )
        .sort("TransactionDT", maintain_order=True)
        .with_columns(time_since_last_txn)
    )


def engineer_features(
    train_lf: pl.LazyFrame, test_lf: pl.LazyFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Fit encoders on train, apply to both splits. Convenience wrapper around
    fit_feature_encoders() + transform_features()."""
    encoders = fit_feature_encoders(train_lf)
    return transform_features(train_lf, encoders), transform_features(test_lf, encoders)
