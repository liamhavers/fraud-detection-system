"""Download and load the raw datasets: Kaggle Credit Card Fraud and IEEE-CIS."""

import pandas as pd

from src.config import (
    CREDITCARD_RAW_DIR,
    IEEE_IDENTITY_FILE,
    IEEE_RAW_DIR,
    IEEE_TRANSACTION_FILE,
    RAW_DATA_FILE,
)


def download_dataset() -> None:
    """Download `mlg-ulb/creditcardfraud` via the Kaggle API into data/raw/creditcard/.

    Requires a Kaggle API token (~/.kaggle/kaggle.json). See README for setup.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "mlg-ulb/creditcardfraud", path=str(CREDITCARD_RAW_DIR), unzip=True
    )


def download_ieee_dataset() -> None:
    """Download the `ieee-fraud-detection` competition files into data/raw/ieee_cis/.

    Requires accepting the competition rules on Kaggle first. See README for setup.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        "ieee-fraud-detection", path=str(IEEE_RAW_DIR), quiet=False
    )

    import zipfile

    zip_path = IEEE_RAW_DIR / "ieee-fraud-detection.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(IEEE_RAW_DIR)
    zip_path.unlink()


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
