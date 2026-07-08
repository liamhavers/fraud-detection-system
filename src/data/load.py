"""Download and load the raw Kaggle Credit Card Fraud dataset."""

import pandas as pd

from src.config import RAW_DATA_FILE


def download_dataset() -> None:
    """Download `mlg-ulb/creditcardfraud` via the Kaggle API into data/raw/.

    Requires a Kaggle API token (~/.kaggle/kaggle.json). See README for setup.
    """
    raise NotImplementedError


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from data/raw/creditcard.csv."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_FILE} not found. Run `download_dataset()` or see README "
            "for manual download instructions."
        )
    return pd.read_csv(RAW_DATA_FILE)
