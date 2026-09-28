"""Download and load the raw datasets: Kaggle Credit Card Fraud and IEEE-CIS."""

import polars as pl

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


def scan_raw_data() -> pl.LazyFrame:
    """Lazily scan the raw Kaggle Credit Card Fraud dataset.

    `Time` is pinned to Float64 rather than inferred: the CSV writes it as a
    plain integer for the first ~150k rows and then switches to scientific
    notation (`1e+05`), which sampled schema inference would type as Int64
    and then fail to parse.
    """
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_FILE} not found. Run `download_dataset()` or see README "
            "for manual download instructions."
        )
    return pl.scan_csv(RAW_DATA_FILE, schema_overrides={"Time": pl.Float64})


def scan_raw_ieee_data() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Lazily scan the raw IEEE-CIS transaction and identity tables (not yet joined).

    Schema inference samples 10k rows rather than polars' default 100: many
    identity/V columns are null for long stretches, and a column typed from
    an all-null sample would come back as String instead of Float64.
    """
    if not IEEE_TRANSACTION_FILE.exists() or not IEEE_IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"IEEE-CIS files not found under {IEEE_TRANSACTION_FILE.parent}. "
            "Run `download_ieee_dataset()` or see README for manual download instructions."
        )
    transaction_lf = pl.scan_csv(IEEE_TRANSACTION_FILE, infer_schema_length=10_000)
    identity_lf = pl.scan_csv(IEEE_IDENTITY_FILE, infer_schema_length=10_000)
    return transaction_lf, identity_lf
