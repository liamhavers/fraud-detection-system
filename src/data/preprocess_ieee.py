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
"""

import pandas as pd

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
    transaction_df: pd.DataFrame, identity_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join transaction and identity tables on TransactionID.

    Most transactions (~76%) have no matching identity row — that's a
    structural pattern worth keeping visible as its own feature rather than
    letting it collapse into "missing", since a transaction with no device/
    identity fingerprint at all is a different situation from one where a
    specific identity field happens to be null.
    """
    joined = transaction_df.merge(identity_df, on="TransactionID", how="left")
    joined["has_identity"] = joined["TransactionID"].isin(identity_df["TransactionID"]).astype(int)
    return joined


def time_aware_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by TransactionDT so training data precedes test data (see preprocess.py)."""
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    return train_df, test_df


def fit_feature_encoders(train_df: pd.DataFrame) -> dict:
    """Fit categorical vocabularies and card1 aggregation stats on TRAIN ONLY.

    Returns a dict of fitted artifacts to be passed to transform_features()
    for both the train and test splits, so test never influences what the
    encoders learned.
    """
    categorical_columns_present = [col for col in CATEGORICAL_COLUMNS if col in train_df.columns]
    category_maps = {
        col: {cat: code for code, cat in enumerate(train_df[col].astype("category").cat.categories)}
        for col in categorical_columns_present
    }

    card1_frequency = train_df["card1"].value_counts()
    card1_mean_amount = train_df.groupby("card1")["TransactionAmt"].mean()
    global_mean_amount = train_df["TransactionAmt"].mean()

    null_fraction = train_df.isnull().mean().sort_values(ascending=False)
    missing_indicator_columns = null_fraction.head(N_MISSING_INDICATOR_COLUMNS).index.tolist()

    return {
        "category_maps": category_maps,
        "card1_frequency": card1_frequency,
        "card1_mean_amount": card1_mean_amount,
        "global_mean_amount": global_mean_amount,
        "missing_indicator_columns": missing_indicator_columns,
    }


def transform_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply encoders fitted by fit_feature_encoders() to a train or test split.

    Missing values in the underlying V/D/C feature blocks are left as NaN
    rather than imputed — XGBoost handles missing values natively by
    learning a default split direction per node, and imputing ~85%-null
    columns with a fabricated value would add noise, not signal. The
    indicator flags below exist for the columns where missingness itself
    looks informative enough to surface explicitly (see EDA notebook).
    """
    df = df.copy()

    for col in encoders["missing_indicator_columns"]:
        df[f"{col}_is_missing"] = df[col].isnull().astype(int)

    for col, mapping in encoders["category_maps"].items():
        df[col] = df[col].map(mapping).fillna(UNSEEN_CATEGORY_CODE).astype(int)

    df["card1_frequency"] = df["card1"].map(encoders["card1_frequency"]).fillna(0)
    df["card1_mean_amount"] = (
        df["card1"].map(encoders["card1_mean_amount"]).fillna(encoders["global_mean_amount"])
    )

    df = df.sort_values("TransactionDT")
    df["time_since_last_txn_same_card"] = (
        df.groupby("card1")["TransactionDT"].diff().fillna(-1)
    )

    return df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit encoders on train, apply to both splits. Convenience wrapper around
    fit_feature_encoders() + transform_features()."""
    encoders = fit_feature_encoders(train_df)
    return transform_features(train_df, encoders), transform_features(test_df, encoders)
