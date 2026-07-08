"""Feature-distribution drift check between a reference window and new data.

Uses Population Stability Index (PSI) per feature; values above
DRIFT_PSI_THRESHOLD flag meaningful distribution shift worth investigating.
"""

import numpy as np
import pandas as pd

from src.config import DRIFT_PSI_THRESHOLD


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Compute PSI between a reference and current feature distribution."""
    raise NotImplementedError


def check_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = DRIFT_PSI_THRESHOLD
) -> pd.DataFrame:
    """Return a per-feature PSI report flagging features that have drifted."""
    raise NotImplementedError
