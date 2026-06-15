"""Tests for synthdrive.claims.frequency."""

import numpy as np
import pandas as pd
import pytest

from synthdrive.claims.frequency import (
    FrequencyModel,
    compute_frequency_risk_score,
    compute_zero_inflation_prob,
    sample_claim_counts,
    _sample_negative_binomial,
)
from synthdrive.presets.core import load_core_preset


@pytest.fixture
def freq_params():
    return load_core_preset().frequency


@pytest.fixture
def sample_feature_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "exposure": rng.uniform(0.25, 1.0, n),
        "insured_age": rng.integers(20, 70, n),
        "insured_sex": np.where(rng.random(n) < 0.539, "Male", "Female"),
        "marital_status": np.where(rng.random(n) < 0.699, "Married", "Single"),
        "credit_score": rng.integers(450, 850, n),
        "years_no_claims": rng.integers(0, 10, n),
        "car_use": np.where(rng.random(n) < 0.96, "Commute", "Commercial"),
        "territory": rng.integers(11, 92, n),          # integer zone codes 11–91
        "annual_miles_drive": rng.lognormal(9.3, 0.4, n),
        "annual_pct_driven": rng.beta(2, 4, n),
        "brake_9_miles": rng.integers(1, 30, n),       # raw event counts
        "accel_9_miles": rng.integers(5, 80, n),       # raw event counts
    })


# ---------------------------------------------------------------------------
# _sample_negative_binomial
# ---------------------------------------------------------------------------


def test_nb_sample_shape():
    rng = np.random.default_rng(1)
    mu = np.full(500, 0.06)
    alpha = 0.80
    counts = _sample_negative_binomial(mu, alpha, rng)
    assert counts.shape == (500,)
    assert counts.dtype == np.int64


def test_nb_sample_non_negative():
    rng = np.random.default_rng(2)
    mu = np.full(1000, 0.1)
    counts = _sample_negative_binomial(mu, 0.5, rng)
    assert (counts >= 0).all()


def test_nb_sample_mean_approx():
    """E[N] should be approximately mu for large samples."""
    rng = np.random.default_rng(3)
    target_mu = 0.10
    mu = np.full(50_000, target_mu)
    counts = _sample_negative_binomial(mu, 0.8, rng)
    sample_mean = counts.mean()
    assert abs(sample_mean - target_mu) < 0.015, \
        f"NB mean {sample_mean:.4f} far from target {target_mu}"


def test_nb_variance_overdispersed():
    """Var[N] should exceed mu (overdispersion)."""
    rng = np.random.default_rng(4)
    mu_val = 0.15
    alpha = 1.0
    mu = np.full(100_000, mu_val)
    counts = _sample_negative_binomial(mu, alpha, rng)
    expected_var = mu_val + alpha * mu_val ** 2
    sample_var = counts.var()
    # Allow ±50% relative tolerance for variance estimation
    assert abs(sample_var - expected_var) / expected_var < 0.5


def test_nb_non_integer_alpha():
    """Non-integer 1/alpha should not raise."""
    rng = np.random.default_rng(5)
    mu = np.full(100, 0.05)
    counts = _sample_negative_binomial(mu, 1.7, rng)  # 1/1.7 is non-integer
    assert (counts >= 0).all()


# ---------------------------------------------------------------------------
# compute_zero_inflation_prob
# ---------------------------------------------------------------------------


def test_zero_inflation_prob_in_0_1(sample_feature_df, freq_params):
    pi = compute_zero_inflation_prob(sample_feature_df, freq_params)
    assert (pi >= 0).all() and (pi <= 0.95).all()


def test_zero_inflation_increases_with_ynclaims(freq_params):
    """More claim-free years → higher structural zero probability."""
    base = {"exposure": [1.0], "credit_score": [650], "insured_age": [40],
            "insured_sex": ["Male"], "marital_status": ["Married"], "car_use": ["Private"],
            "territory": [50], "annual_miles_drive": [12000],
            "annual_pct_driven": [0.33], "brake_9_miles": [6],
            "accel_9_miles": [24]}

    df_low = pd.DataFrame({**base, "years_no_claims": [0]})
    df_high = pd.DataFrame({**base, "years_no_claims": [15]})

    pi_low = compute_zero_inflation_prob(df_low, freq_params)[0]
    pi_high = compute_zero_inflation_prob(df_high, freq_params)[0]
    assert pi_high > pi_low


# ---------------------------------------------------------------------------
# compute_frequency_risk_score
# ---------------------------------------------------------------------------


def test_risk_score_young_driver_beta_applied(freq_params):
    # beta_young is calibrated negative to offset copula-driven feature excess
    # for young drivers. The raw score term is therefore lower for age<25, while
    # aggregate frequency is matched via the seed-mode feature correlations.
    base = {
        "insured_age": [45], "insured_sex": ["Female"], "marital_status": ["Married"],
        "credit_score": [650], "years_no_claims": [3], "car_use": ["Private"],
        "territory": [50], "annual_miles_drive": [12000],
        "annual_pct_driven": [0.33], "brake_9_miles": [6],
        "accel_9_miles": [24], "exposure": [1.0],
    }
    df_young = pd.DataFrame({**base, "insured_age": [20]})
    df_old = pd.DataFrame({**base})
    score_young = compute_frequency_risk_score(df_young, freq_params)[0]
    score_old = compute_frequency_risk_score(df_old, freq_params)[0]
    # The difference should equal beta_young exactly (positive or negative).
    assert abs((score_young - score_old) - freq_params.beta_young) < 1e-9


def test_risk_score_commercial_higher(freq_params):
    base = {
        "insured_age": [40], "insured_sex": ["Female"], "marital_status": ["Married"],
        "credit_score": [650], "years_no_claims": [3], "territory": [50],
        "annual_miles_drive": [12000], "annual_pct_driven": [0.33],
        "brake_9_miles": [6], "accel_9_miles": [24], "exposure": [1.0],
    }
    df_commercial = pd.DataFrame({**base, "car_use": ["Commercial"]})
    df_private = pd.DataFrame({**base, "car_use": ["Private"]})
    score_c = compute_frequency_risk_score(df_commercial, freq_params)[0]
    score_p = compute_frequency_risk_score(df_private, freq_params)[0]
    assert score_c > score_p


# ---------------------------------------------------------------------------
# sample_claim_counts
# ---------------------------------------------------------------------------


def test_sample_claim_counts_shape(sample_feature_df, freq_params):
    rng = np.random.default_rng(10)
    counts = sample_claim_counts(sample_feature_df, freq_params, rng)
    assert counts.shape == (len(sample_feature_df),)
    assert counts.dtype == np.int64


def test_sample_claim_counts_non_negative(sample_feature_df, freq_params):
    rng = np.random.default_rng(11)
    counts = sample_claim_counts(sample_feature_df, freq_params, rng)
    assert (counts >= 0).all()


def test_frequency_model_class(sample_feature_df, freq_params):
    model = FrequencyModel(freq_params)
    rng = np.random.default_rng(12)
    counts = model.sample(sample_feature_df, rng)
    assert len(counts) == len(sample_feature_df)
    assert (counts >= 0).all()


def test_expected_frequency_positive(sample_feature_df, freq_params):
    model = FrequencyModel(freq_params)
    ef = model.expected_frequency(sample_feature_df)
    assert (ef > 0).all()
