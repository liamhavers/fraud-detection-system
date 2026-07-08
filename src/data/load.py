"""Download and load the raw datasets: Kaggle Credit Card Fraud and IEEE-CIS."""

import pandas as pd

from src.config import IEEE_IDENTITY_FILE, IEEE_TRANSACTION_FILE, RAW_DATA_FILE


def download_dataset() -> None:
    """Download `mlg-ulb/creditcardfraud` via the Kaggle API into data/raw/creditcard/.

    Requires a Kaggle API token (~/.kaggle/kaggle.json). See README for setup.
    """
    raise NotImplementedError


def download_ieee_dataset() -> None:
    """Download the `ieee-fraud-detection` competition files into data/raw/ieee_cis/.

    Requires accepting the competition rules on Kaggle first. See README for setup.
    """
    raise NotImplementedError


def load_raw_data() -> pd.DataFrame:
    """Load the raw Kaggle Credit Card Fraud dataset."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_FILE} not found. Run `download_dataset()` or see README "
            "for manual download instructions."
        )
    return pd.read_csv(RAW_DATA_FILE)


def load_raw_ieee_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw IEEE-CIS transaction and identity tables (not yet joined)."""
    if not IEEE_TRANSACTION_FILE.exists() or not IEEE_IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"IEEE-CIS files not found under {IEEE_TRANSACTION_FILE.parent}. "
            "Run `download_ieee_dataset()` or see README for manual download instructions."
        )
    transaction_df = pd.read_csv(IEEE_TRANSACTION_FILE)
    identity_df = pd.read_csv(IEEE_IDENTITY_FILE)
    return transaction_df, identity_df
