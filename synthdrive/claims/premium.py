"""
Pure premium computation for SynthDrive v0.1.

Pure premium is the expected annual loss cost:

    pure_premium_i = claim_amount_i / exposure_i

For policies with exposure = 0 (should not occur after constraint enforcement
but handled defensively), pure_premium is set to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pure_premium(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pure premium as claim_amount / exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "claim_amount" and "exposure" columns.

    Returns
    -------
    ndarray of float64, shape (n,)
    """
    if "claim_amount" not in df.columns:
        raise ValueError("df must contain 'claim_amount'.")
    if "exposure" not in df.columns:
        raise ValueError("df must contain 'exposure'.")

    claim_amount = df["claim_amount"].values.astype(float)
    exposure = df["exposure"].values.astype(float)

    # Guard against zero exposure
    safe_exposure = np.where(exposure > 0.0, exposure, np.nan)
    pp = claim_amount / safe_exposure
    pp = np.where(np.isfinite(pp), pp, 0.0)
    pp = np.maximum(pp, 0.0)

    return pp
