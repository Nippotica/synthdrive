"""
Gamma claim severity model for SynthDrive v0.1.

Model structure
---------------
For policy i with N_i > 0 claims and risk covariates X_i:

    A_i | N_i = n, X_i ~ Gamma(shape = shape_per_claim * n,
                                scale = scale_base * sev_factor(X_i))

where:
    sev_factor(X_i) = exp(sum of applicable beta coefficients)
    E[A_i | N_i = n, X_i] = n * shape_per_claim * scale_base * sev_factor(X_i)

For the reference driver with N=1 claim:
    E[A] = shape_per_claim * scale_base = 2.0 * 1500 = 3 000 monetary units

Policies with N_i = 0 receive A_i = 0 unconditionally.

Post-sampling
-------------
- A minimum claim threshold of min_claim_amount is enforced for N > 0.
- The threshold is not zero to avoid unrealistic micro-claims distorting
  GLM fits downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Severity factor (multiplicative adjustment on scale)
# ---------------------------------------------------------------------------


def compute_severity_factor(
    df: pd.DataFrame,
    params: object,
) -> np.ndarray:
    """
    Compute the multiplicative severity adjustment factor for each policy.

        sev_factor_i = exp(sum of applicable log-RR coefficients)

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    params : SeverityParams

    Returns
    -------
    ndarray of float64, shape (n,)
    """
    n = len(df)
    log_factor = np.zeros(n, dtype=float)

    # Young driver
    if "insured_age" in df.columns:
        age = df["insured_age"].values.astype(float)
        log_factor += params.beta_young * (age < 25).astype(float)

    # Log annual miles (centered at log(12 000))
    if "annual_miles_drive" in df.columns:
        miles = np.maximum(df["annual_miles_drive"].values.astype(float), 1.0)
        log_factor += params.beta_log_miles * (np.log(miles) - np.log(12_000.0))

    # Car age (centered at 7 years)
    if "car_age" in df.columns:
        car_age = df["car_age"].values.astype(float)
        log_factor += params.beta_car_age * (car_age - 7.0)

    return np.exp(log_factor)


# ---------------------------------------------------------------------------
# Core sampling function
# ---------------------------------------------------------------------------


def sample_aggregate_severity(
    df: pd.DataFrame,
    claim_counts: np.ndarray,
    params: object,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample aggregate claim amounts for each policy.

    Policies with claim_count == 0 receive amount = 0.
    Policies with claim_count > 0 receive a Gamma sample.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    claim_counts : ndarray of int64, shape (n,)
        Claim counts from the frequency model.
    params : SeverityParams
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of float64, shape (n,) — aggregate claim amounts.
    """
    n = len(df)
    amounts = np.zeros(n, dtype=float)

    claim_mask = claim_counts > 0
    n_claims = int(claim_mask.sum())
    if n_claims == 0:
        return amounts

    # Parameters for claimant sub-population
    counts_c = claim_counts[claim_mask].astype(float)
    sev_factor = compute_severity_factor(df, params)
    sev_factor_c = sev_factor[claim_mask]

    # Gamma parameterisation:
    #   shape = shape_per_claim * N   (increases with claim count)
    #   scale = scale_base * sev_factor (adjusts mean severity)
    gamma_shape = params.shape_per_claim * counts_c          # array
    gamma_scale = params.scale_base * sev_factor_c           # array

    # Vectorised gamma sampling
    raw_amounts = rng.gamma(shape=gamma_shape, scale=gamma_scale)

    # Enforce minimum claim threshold for policies with N > 0
    min_threshold = getattr(params, "min_claim_amount", 100.0)
    raw_amounts = np.maximum(raw_amounts, min_threshold)

    amounts[claim_mask] = raw_amounts
    return amounts


# ---------------------------------------------------------------------------
# SeverityModel class (thin wrapper)
# ---------------------------------------------------------------------------


class SeverityModel:
    """
    Wraps gamma severity sampling for use in the generate pipeline.

    Parameters
    ----------
    params : SeverityParams
    """

    def __init__(self, params: object) -> None:
        self.params = params

    def sample(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        claim_counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return aggregate claim amount for each row.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain "claim_count" column if claim_counts is None.
        rng : numpy.random.Generator
        claim_counts : ndarray, optional
            If provided, used instead of df["claim_count"].

        Returns
        -------
        ndarray of float64, shape (n,)
        """
        if claim_counts is None:
            if "claim_count" not in df.columns:
                raise ValueError(
                    "claim_counts must be provided or df must contain 'claim_count'."
                )
            claim_counts = df["claim_count"].values.astype(np.int64)

        return sample_aggregate_severity(df, claim_counts, self.params, rng)

    def expected_severity(
        self,
        df: pd.DataFrame,
        n_claims: int = 1,
    ) -> np.ndarray:
        """
        Return E[A | N=n_claims, X] = n * shape_per_claim * scale_base * sev_factor(X).
        Useful for diagnostics.
        """
        sev_factor = compute_severity_factor(df, self.params)
        expected = (
            n_claims * self.params.shape_per_claim
            * self.params.scale_base
            * sev_factor
        )
        return expected


# Avoid NameError in SeverityModel.sample
from typing import Optional
