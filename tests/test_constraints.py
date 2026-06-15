"""Tests for synthdrive.features.constraints."""

import numpy as np
import pandas as pd
import pytest

from synthdrive.features.constraints import (
    all_passed,
    check_constraints,
    enforce_all_constraints,
    enforce_claim_consistency,
    enforce_compositional_dow,
    enforce_exposure,
    enforce_monotone_thresholds,
    format_constraint_report,
    recompute_pure_premium,
)
from synthdrive.data.schema import ACCEL_COLUMNS, DOW_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_df(n=10):
    """Return a minimal DataFrame covering all required columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "duration": rng.integers(91, 366, n),        # days
        "exposure": rng.uniform(0.25, 1.0, n),
        "insured_age": rng.integers(20, 70, n),
        "credit_score": rng.integers(500, 800, n),
        "years_no_claims": rng.integers(0, 10, n),
        "car_age": rng.integers(-2, 20, n),           # range [-2, 20]
        "annual_miles_drive": rng.uniform(5000, 20000, n),
        "annual_pct_driven": rng.uniform(0.1, 0.9, n),
        "total_miles_driven": rng.uniform(1000, 15000, n),
        "avg_days_week": rng.uniform(2, 7, n),
        "pct_drive_mon": rng.uniform(0.1, 0.2, n),
        "pct_drive_tue": rng.uniform(0.1, 0.2, n),
        "pct_drive_wed": rng.uniform(0.1, 0.2, n),
        "pct_drive_thu": rng.uniform(0.1, 0.2, n),
        "pct_drive_fri": rng.uniform(0.1, 0.2, n),
        "pct_drive_sat": rng.uniform(0.05, 0.15, n),
        "pct_drive_sun": rng.uniform(0.05, 0.10, n),
        "accel_6_miles": rng.integers(50, 200, n),   # raw event counts
        "accel_8_miles": rng.integers(25, 100, n),
        "accel_9_miles": rng.integers(10, 60, n),
        "accel_11_miles": rng.integers(5, 30, n),    # no accel_10_miles
        "accel_12_miles": rng.integers(2, 15, n),
        "accel_14_miles": rng.integers(0, 6, n),     # added accel_14_miles
        "brake_6_miles": rng.integers(30, 150, n),
        "brake_8_miles": rng.integers(15, 80, n),
        "brake_9_miles": rng.integers(5, 40, n),
        "brake_11_miles": rng.integers(2, 20, n),    # no brake_10_miles
        "brake_12_miles": rng.integers(1, 10, n),
        "brake_14_miles": rng.integers(0, 4, n),     # added brake_14_miles
        "left_turn_intensity_08": rng.integers(200, 2000, n),  # raw counts
        "left_turn_intensity_09": rng.integers(100, 1000, n),
        "left_turn_intensity_10": rng.integers(50, 500, n),
        "left_turn_intensity_11": rng.integers(20, 200, n),
        "left_turn_intensity_12": rng.integers(5, 80, n),
        "right_turn_intensity_08": rng.integers(300, 3000, n),
        "right_turn_intensity_09": rng.integers(150, 1500, n),
        "right_turn_intensity_10": rng.integers(60, 600, n),
        "right_turn_intensity_11": rng.integers(25, 250, n),
        "right_turn_intensity_12": rng.integers(8, 100, n),
        "claim_count": rng.integers(0, 3, n),
        "claim_amount": rng.uniform(0, 5000, n),
        "pure_premium": rng.uniform(0, 5000, n),
    })


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


def test_enforce_exposure_clips_to_positive():
    df = pd.DataFrame({"exposure": [-0.1, 0.0, 0.5, 1.5], "duration": [91, 182, 365, 366]})
    out = enforce_exposure(df)
    assert (out["exposure"] > 0).all()
    assert (out["exposure"] <= 1.0).all()
    # Duration (in days) is not clipped by enforce_exposure
    assert (out["duration"] == df["duration"]).all()


# ---------------------------------------------------------------------------
# Compositional DOW
# ---------------------------------------------------------------------------


def test_compositional_dow_sums_to_one():
    n = 50
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.05, 0.25, (n, 7))
    # Do NOT normalize — let enforce_compositional_dow do it
    df = pd.DataFrame(raw, columns=list(DOW_COLUMNS))
    out = enforce_compositional_dow(df)
    sums = out[list(DOW_COLUMNS)].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


def test_compositional_dow_wkday_wkend_sum_to_one():
    n = 20
    rng = np.random.default_rng(2)
    raw = rng.dirichlet(np.ones(7), size=n)
    df = pd.DataFrame(raw, columns=list(DOW_COLUMNS))
    out = enforce_compositional_dow(df)
    total = out["pct_drive_wkday"] + out["pct_drive_wkend"]
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_compositional_dow_zero_row_becomes_uniform():
    df = pd.DataFrame({d: [0.0] for d in DOW_COLUMNS})
    out = enforce_compositional_dow(df)
    np.testing.assert_allclose(out[list(DOW_COLUMNS)].values, 1/7, atol=1e-9)


# ---------------------------------------------------------------------------
# Monotone threshold sequences
# ---------------------------------------------------------------------------


def test_monotone_thresholds_enforced():
    n = 30
    rng = np.random.default_rng(3)
    # Scramble the order so monotonicity is violated
    df = pd.DataFrame(
        rng.uniform(0, 1, (n, len(ACCEL_COLUMNS))),
        columns=list(ACCEL_COLUMNS),
    )
    out = enforce_monotone_thresholds(df, ACCEL_COLUMNS)
    arr = out[list(ACCEL_COLUMNS)].values
    for i in range(arr.shape[1] - 1):
        assert (arr[:, i] >= arr[:, i + 1] - 1e-9).all(), \
            f"Monotone violated between col {i} and {i+1}"


# ---------------------------------------------------------------------------
# Claim consistency
# ---------------------------------------------------------------------------


def test_claim_amount_zero_when_count_zero():
    df = pd.DataFrame({
        "claim_count": [0, 0, 1, 2, 0],
        "claim_amount": [500.0, 200.0, 1000.0, 3000.0, 100.0],
    })
    out = enforce_claim_consistency(df)
    zero_mask = out["claim_count"] == 0
    assert (out.loc[zero_mask, "claim_amount"] == 0.0).all()
    assert out.loc[2, "claim_amount"] == 1000.0  # unchanged
    assert out.loc[3, "claim_amount"] == 3000.0


def test_claim_count_non_negative():
    df = pd.DataFrame({"claim_count": [-1, -5, 0, 2], "claim_amount": [0.0]*4})
    out = enforce_claim_consistency(df)
    assert (out["claim_count"] >= 0).all()


# ---------------------------------------------------------------------------
# Pure premium
# ---------------------------------------------------------------------------


def test_pure_premium_computation():
    df = pd.DataFrame({
        "claim_amount": [0.0, 3000.0, 6000.0],
        "exposure": [1.0, 0.5, 1.0],
    })
    pp = recompute_pure_premium(df)["pure_premium"].values
    np.testing.assert_allclose(pp, [0.0, 6000.0, 6000.0])


def test_pure_premium_zero_exposure_handled():
    df = pd.DataFrame({"claim_amount": [1000.0], "exposure": [0.0]})
    out = recompute_pure_premium(df)
    assert out["pure_premium"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# enforce_all_constraints (integration)
# ---------------------------------------------------------------------------


def test_enforce_all_constraints_idempotent(small_df):
    """Applying constraints twice gives the same result as once."""
    out1 = enforce_all_constraints(small_df)
    out2 = enforce_all_constraints(out1)
    pd.testing.assert_frame_equal(out1.reset_index(drop=True),
                                   out2.reset_index(drop=True))


def test_check_constraints_all_pass(small_df):
    results = check_constraints(small_df)
    report = format_constraint_report(results)
    assert all_passed(results), f"Constraint failures:\n{report}"


# ---------------------------------------------------------------------------
# format_constraint_report
# ---------------------------------------------------------------------------


def test_format_constraint_report_contains_pass_fail():
    results = [("check_a", True, "OK"), ("check_b", False, "2/10 rows fail.")]
    report = format_constraint_report(results)
    assert "FAIL" in report
    assert "check_b" in report
    assert "check_a" in report
