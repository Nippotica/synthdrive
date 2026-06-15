"""Tests for synthdrive.claims.severity."""

import numpy as np
import pandas as pd
import pytest

from synthdrive.claims.severity import (
    SeverityModel,
    compute_severity_factor,
    sample_aggregate_severity,
)
from synthdrive.presets.core import load_core_preset


@pytest.fixture
def sev_params():
    return load_core_preset().severity


@pytest.fixture
def feature_df():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "insured_age": rng.integers(20, 70, n),
        "car_age": rng.integers(0, 15, n),
        "annual_miles_drive": rng.lognormal(9.3, 0.4, n),
        "exposure": rng.uniform(0.25, 1.0, n),
    })


# ---------------------------------------------------------------------------
# compute_severity_factor
# ---------------------------------------------------------------------------


def test_severity_factor_positive(feature_df, sev_params):
    factors = compute_severity_factor(feature_df, sev_params)
    assert (factors > 0).all()


def test_severity_factor_young_driver_higher(sev_params):
    base = {"car_age": [7], "annual_miles_drive": [12000], "exposure": [1.0]}
    df_young = pd.DataFrame({**base, "insured_age": [20]})
    df_old = pd.DataFrame({**base, "insured_age": [50]})
    f_young = compute_severity_factor(df_young, sev_params)[0]
    f_old = compute_severity_factor(df_old, sev_params)[0]
    assert f_young > f_old


# ---------------------------------------------------------------------------
# sample_aggregate_severity
# ---------------------------------------------------------------------------


def test_severity_zero_for_nonclaimants(feature_df, sev_params):
    n = len(feature_df)
    claim_counts = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(1)
    amounts = sample_aggregate_severity(feature_df, claim_counts, sev_params, rng)
    assert (amounts == 0.0).all()


def test_severity_positive_for_claimants(feature_df, sev_params):
    n = len(feature_df)
    claim_counts = np.ones(n, dtype=np.int64)  # every policy has 1 claim
    rng = np.random.default_rng(2)
    amounts = sample_aggregate_severity(feature_df, claim_counts, sev_params, rng)
    assert (amounts > 0).all()


def test_severity_above_minimum_threshold(feature_df, sev_params):
    n = len(feature_df)
    claim_counts = np.ones(n, dtype=np.int64)
    rng = np.random.default_rng(3)
    amounts = sample_aggregate_severity(feature_df, claim_counts, sev_params, rng)
    assert (amounts >= sev_params.min_claim_amount).all()


def test_severity_shape(feature_df, sev_params):
    n = len(feature_df)
    claim_counts = np.ones(n, dtype=np.int64)
    rng = np.random.default_rng(4)
    amounts = sample_aggregate_severity(feature_df, claim_counts, sev_params, rng)
    assert amounts.shape == (n,)


def test_severity_mean_approx(sev_params):
    """E[A | N=1] should be close to shape_per_claim * scale_base."""
    n = 50_000
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "insured_age": np.full(n, 45),
        "car_age": np.full(n, 7),
        "annual_miles_drive": np.full(n, 12_000.0),
        "exposure": np.ones(n),
    })
    claim_counts = np.ones(n, dtype=np.int64)
    amounts = sample_aggregate_severity(df, claim_counts, sev_params, rng)
    expected = sev_params.shape_per_claim * sev_params.scale_base
    sample_mean = amounts.mean()
    assert abs(sample_mean - expected) / expected < 0.05, \
        f"Severity mean {sample_mean:.0f} far from expected {expected:.0f}"


def test_severity_doubles_with_two_claims(sev_params):
    """E[A | N=2] ≈ 2 * E[A | N=1]."""
    n = 50_000
    rng1 = np.random.default_rng(6)
    rng2 = np.random.default_rng(6)
    df = pd.DataFrame({
        "insured_age": np.full(n, 45),
        "car_age": np.full(n, 7),
        "annual_miles_drive": np.full(n, 12_000.0),
        "exposure": np.ones(n),
    })
    a1 = sample_aggregate_severity(df, np.ones(n, dtype=np.int64), sev_params, rng1)
    a2 = sample_aggregate_severity(df, np.full(n, 2, dtype=np.int64), sev_params, rng2)
    ratio = a2.mean() / a1.mean()
    assert abs(ratio - 2.0) < 0.10, f"Expected mean ratio ≈ 2, got {ratio:.3f}"


# ---------------------------------------------------------------------------
# SeverityModel class
# ---------------------------------------------------------------------------


def test_severity_model_class(feature_df, sev_params):
    n = len(feature_df)
    feature_df = feature_df.copy()
    feature_df["claim_count"] = np.where(
        np.arange(n) % 5 == 0, 1, 0
    ).astype(np.int64)
    model = SeverityModel(sev_params)
    rng = np.random.default_rng(7)
    amounts = model.sample(feature_df, rng)
    assert amounts.shape == (n,)
    assert (amounts >= 0).all()
    zero_mask = feature_df["claim_count"] == 0
    assert (amounts[zero_mask.values] == 0.0).all()


def test_expected_severity_positive(feature_df, sev_params):
    model = SeverityModel(sev_params)
    es = model.expected_severity(feature_df, n_claims=1)
    assert (es > 0).all()
