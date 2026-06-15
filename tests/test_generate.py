"""Tests for synthdrive.generate."""

import numpy as np
import pandas as pd
import pytest

import synthdrive
from synthdrive.data.schema import CANONICAL_COLUMNS
from synthdrive.features.constraints import all_passed, check_constraints


# ---------------------------------------------------------------------------
# Basic shape and column tests
# ---------------------------------------------------------------------------


def test_generate_returns_dataframe():
    df = synthdrive.generate(n=100, random_state=1)
    assert isinstance(df, pd.DataFrame)


def test_generate_correct_row_count():
    for n in (1, 50, 500):
        df = synthdrive.generate(n=n, random_state=0)
        assert len(df) == n, f"Expected {n} rows, got {len(df)}"


def test_generate_has_canonical_columns(small_df):
    # All canonical columns (except policy_id which is prepended) must be present
    canonical_no_id = [c for c in CANONICAL_COLUMNS if c != "policy_id"]
    for col in canonical_no_id:
        assert col in small_df.columns, f"Missing column: {col}"


def test_generate_has_policy_id(small_df):
    assert "policy_id" in small_df.columns
    assert small_df["policy_id"].iloc[0] == "P00000001"
    assert small_df["policy_id"].nunique() == len(small_df)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_generate_reproducible():
    df1 = synthdrive.generate(n=200, random_state=7)
    df2 = synthdrive.generate(n=200, random_state=7)
    pd.testing.assert_frame_equal(
        df1.drop(columns=["policy_id"]),
        df2.drop(columns=["policy_id"]),
    )


def test_generate_different_seeds_differ():
    df1 = synthdrive.generate(n=200, random_state=1)
    df2 = synthdrive.generate(n=200, random_state=2)
    assert not df1["insured_age"].equals(df2["insured_age"])


# ---------------------------------------------------------------------------
# Constraint compliance
# ---------------------------------------------------------------------------


def test_generate_passes_all_constraints(small_df):
    results = check_constraints(small_df)
    assert all_passed(results), (
        "Generated dataset failed constraint checks:\n"
        + "\n".join(f"  FAIL {n}: {m}" for n, p, m in results if not p)
    )


# ---------------------------------------------------------------------------
# Proportion conventions (Q2: all pct_ in [0, 1])
# ---------------------------------------------------------------------------


def test_proportion_columns_in_0_1(small_df):
    pct_cols = [c for c in small_df.columns if c.startswith("pct_")]
    for col in pct_cols:
        vals = small_df[col].values
        assert (vals >= 0).all() and (vals <= 1).all(), \
            f"{col} has values outside [0, 1]"


def test_dow_sums_to_one(small_df):
    from synthdrive.data.schema import DOW_COLUMNS
    dow_sums = small_df[list(DOW_COLUMNS)].sum(axis=1)
    np.testing.assert_allclose(dow_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Claim plausibility
# ---------------------------------------------------------------------------


def test_claim_frequency_plausible(medium_df):
    """Frequency should be in a sensible range for personal auto."""
    total_exp = medium_df["exposure"].sum()
    total_claims = medium_df["claim_count"].sum()
    freq = total_claims / total_exp
    assert 0.01 <= freq <= 0.30, f"Implausible claim frequency: {freq:.4f}"


def test_zero_claim_pct_plausible(medium_df):
    zero_pct = (medium_df["claim_count"] == 0).mean()
    assert 0.70 <= zero_pct <= 0.99, f"Implausible zero-claim pct: {zero_pct:.3f}"


def test_claim_amounts_positive_for_claimants(small_df):
    claimants = small_df[small_df["claim_count"] > 0]
    if len(claimants) > 0:
        assert (claimants["claim_amount"] > 0).all()


def test_claim_amount_zero_for_nonclaimants(small_df):
    non_claimants = small_df[small_df["claim_count"] == 0]
    assert (non_claimants["claim_amount"] == 0.0).all()


def test_pure_premium_consistent(small_df):
    """pure_premium = claim_amount / exposure for each row."""
    df = small_df.copy()
    computed = df["claim_amount"] / df["exposure"]
    # For zero amounts, both should be zero
    zero_mask = df["claim_amount"] == 0
    assert (df.loc[zero_mask, "pure_premium"] == 0.0).all()
    # For non-zero, check agreement
    nonzero_mask = ~zero_mask
    if nonzero_mask.sum() > 0:
        np.testing.assert_allclose(
            df.loc[nonzero_mask, "pure_premium"].values,
            computed[nonzero_mask].values,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Missing seed path handling (Q1: parameter mode fallback)
# ---------------------------------------------------------------------------


def test_generate_without_seed_path():
    """Passing seed_path=None must not raise."""
    df = synthdrive.generate(n=50, seed_path=None, random_state=0)
    assert len(df) == 50


def test_generate_with_nonexistent_seed_path(tmp_path):
    """Passing a nonexistent seed_path must warn and fall back, not raise."""
    import warnings
    nonexistent = str(tmp_path / "does_not_exist.csv")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = synthdrive.generate(n=50, seed_path=nonexistent, random_state=0)
    assert len(df) == 50
    assert any("seed" in str(warning.message).lower() or
               "parameter" in str(warning.message).lower()
               for warning in w)


# ---------------------------------------------------------------------------
# n validation
# ---------------------------------------------------------------------------


def test_generate_raises_on_n_zero():
    with pytest.raises(ValueError, match="n must be >= 1"):
        synthdrive.generate(n=0)


def test_generate_raises_on_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        synthdrive.generate(n=10, preset="nonexistent_preset")
