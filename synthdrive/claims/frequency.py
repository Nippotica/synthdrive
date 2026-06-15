"""
Zero-inflated negative binomial (ZINB) claim frequency model.

Model structure
---------------
For policy i with exposure e_i and risk covariates X_i:

    N_i | not_structural_zero_i ~ NegBin(mu_i, alpha)
    P(structural_zero_i) = pi_i = expit(gamma_0 + gamma * X_i)

where:
    log(mu_i) = log(e_i) + log(base_rate) + risk_score(X_i)
    Var[N_i | NB] = mu_i + alpha * mu_i^2

Sampling
--------
NB with non-integer dispersion is sampled via the gamma–Poisson mixture:
    lambda_i ~ Gamma(shape = 1/alpha, scale = alpha * mu_i)
    N_i      ~ Poisson(lambda_i)

This is exact (not an approximation) and avoids the integer restriction on
numpy's negative_binomial Generator method.

References
----------
Cameron, A. C., and Trivedi, P. K. (2013). Regression Analysis of Count Data,
2nd ed. Cambridge University Press.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------


def compute_frequency_risk_score(
    df: pd.DataFrame,
    params: object,
) -> np.ndarray:
    """
    Compute the log-relative-risk score for each policy.

    This additive score is applied to log(mu) before exponentiation:
        log(mu_i) = log(exposure_i) + log(base_rate) + risk_score_i

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (must not contain claim columns yet).
    params : FrequencyParams
        Frequency model parameters.

    Returns
    -------
    ndarray of shape (n,) — risk scores on the log scale.
    """
    n = len(df)
    score = np.zeros(n, dtype=float)

    # --- Age effects
    if "insured_age" in df.columns:
        age = df["insured_age"].values.astype(float)
        score += params.beta_young * (age < 25).astype(float)
        score += params.beta_senior * (age > 65).astype(float)

    # --- Sex
    if "insured_sex" in df.columns:
        sex = df["insured_sex"].astype(str).values
        score += params.beta_male * (sex == "Male").astype(float)

    # --- Marital status
    if "marital_status" in df.columns:
        marital = df["marital_status"].astype(str).values
        score += params.beta_single * (marital == "Single").astype(float)

    # --- Credit score (centered at 650, scaled per 100 points)
    if "credit_score" in df.columns:
        credit_norm = (df["credit_score"].values.astype(float) - 650.0) / 100.0
        score += params.beta_credit_100pts * credit_norm

    # --- Car use
    if "car_use" in df.columns:
        car_use = df["car_use"].astype(str).values
        score += params.beta_commercial * (car_use == "Commercial").astype(float)
        score += params.beta_farmer_artisan * (car_use == "Farmer").astype(float)

    # --- Territory: linear function of numeric zone code
    if "territory" in df.columns:
        territory = pd.to_numeric(df["territory"], errors="coerce").fillna(
            params.territory_center
        ).values
        score += params.territory_beta * (territory - params.territory_center)

    # --- Vehicle age (centered at median ≈ 9 years)
    if "car_age" in df.columns:
        car_age = df["car_age"].values.astype(float)
        score += params.beta_car_age * (car_age - 9.0)

    # --- Annual miles (log-centered at log(12 000))
    if "annual_miles_drive" in df.columns:
        miles = df["annual_miles_drive"].values.astype(float)
        miles = np.maximum(miles, 1.0)  # safety guard before log
        log_miles_centered = np.log(miles) - np.log(12_000.0)
        score += params.beta_log_miles * log_miles_centered

    # --- Annual pct driven (centered at 0.33)
    if "annual_pct_driven" in df.columns:
        pct = df["annual_pct_driven"].values.astype(float)
        score += params.beta_pct_driven * (pct - 0.33)

    # --- Hard braking (log1p-transformed, centered at log1p of SBV median)
    if "brake_9_miles" in df.columns:
        brake = df["brake_9_miles"].values.astype(float)
        score += params.beta_hard_brake * (np.log1p(brake) - params.brake_9_center)

    # --- Hard acceleration (log1p-transformed, centered at log1p of SBV median)
    if "accel_9_miles" in df.columns:
        accel = df["accel_9_miles"].values.astype(float)
        score += params.beta_hard_accel * (np.log1p(accel) - params.accel_9_center)

    return score


# ---------------------------------------------------------------------------
# Zero-inflation probability
# ---------------------------------------------------------------------------


def compute_zero_inflation_prob(
    df: pd.DataFrame,
    params: object,
) -> np.ndarray:
    """
    Compute the structural zero-inflation probability for each policy.

    pi_i = expit(gamma_0
                 + gamma_ynclaims * years_no_claims_i
                 + gamma_credit * credit_norm_i)

    Returns
    -------
    ndarray of shape (n,) with values in [0, 1).
    """
    n = len(df)
    logit_pi = np.full(n, params.gamma_0, dtype=float)

    if "years_no_claims" in df.columns:
        yncl = df["years_no_claims"].values.astype(float)
        logit_pi += params.gamma_ynclaims * yncl

    if "credit_score" in df.columns:
        credit_norm = (df["credit_score"].values.astype(float) - 650.0) / 100.0
        logit_pi += params.gamma_credit * credit_norm

    return np.clip(expit(logit_pi), 0.0, 0.95)


# ---------------------------------------------------------------------------
# NB / ZINB sampling via gamma–Poisson mixture
# ---------------------------------------------------------------------------


def _sample_negative_binomial(
    mu: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from NegBin(mu, alpha) via the gamma–Poisson mixture.

        lambda ~ Gamma(shape = 1/alpha, scale = alpha * mu)
        N      ~ Poisson(lambda)

    This supports non-integer 1/alpha and is vectorised over the mu array.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Expected count for each policy (after exposure offset).
    alpha : float
        Overdispersion parameter.  Var[N] = mu + alpha * mu^2.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of int64, shape (n,)
    """
    n = len(mu)
    r = 1.0 / alpha                          # gamma shape (can be non-integer)
    gamma_rates = rng.gamma(shape=r, scale=alpha * mu, size=n)
    gamma_rates = np.maximum(gamma_rates, 1e-10)  # guard Poisson against 0
    counts = rng.poisson(lam=gamma_rates)
    return counts.astype(np.int64)


def sample_claim_counts(
    df: pd.DataFrame,
    params: object,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample ZINB claim counts for a feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with exposure and risk covariates.
    params : FrequencyParams
        Frequency model parameters.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of int64, shape (n,) — claim counts per policy.
    """
    n = len(df)

    # --- Expected count (mu)
    risk_score = compute_frequency_risk_score(df, params)

    if "exposure" in df.columns:
        log_exposure = np.log(np.maximum(df["exposure"].values.astype(float), 1e-8))
    else:
        log_exposure = 0.0

    log_mu = log_exposure + np.log(params.base_rate) + risk_score
    mu = np.exp(log_mu)
    mu = np.clip(mu, 1e-8, 20.0)  # practical upper cap

    # --- Negative binomial counts
    nb_counts = _sample_negative_binomial(mu=mu, alpha=params.alpha, rng=rng)

    # --- Zero inflation
    pi = compute_zero_inflation_prob(df, params)
    structural_zeros = rng.random(n) < pi

    counts = np.where(structural_zeros, 0, nb_counts).astype(np.int64)
    return counts


# ---------------------------------------------------------------------------
# FrequencyModel class (thin wrapper)
# ---------------------------------------------------------------------------


class FrequencyModel:
    """
    Wraps ZINB frequency sampling for use in the generate pipeline.

    Parameters
    ----------
    params : FrequencyParams
    """

    def __init__(self, params: object) -> None:
        self.params = params

    def sample(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Return claim count array for each row in df.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame.
        rng : numpy.random.Generator

        Returns
        -------
        ndarray of int64, shape (n,)
        """
        return sample_claim_counts(df, self.params, rng)

    def expected_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return E[N_i] = (1 - pi_i) * mu_i for each policy.

        Useful for diagnostics without sampling.
        """
        risk_score = compute_frequency_risk_score(df, self.params)
        if "exposure" in df.columns:
            log_exposure = np.log(np.maximum(df["exposure"].values.astype(float), 1e-8))
        else:
            log_exposure = 0.0

        mu = np.exp(log_exposure + np.log(self.params.base_rate) + risk_score)
        mu = np.clip(mu, 1e-8, 20.0)
        pi = compute_zero_inflation_prob(df, self.params)
        return (1.0 - pi) * mu
