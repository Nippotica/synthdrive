"""
Constraint enforcement for SynthDrive v0.1.

This module enforces actuarial and physical constraints on generated datasets.
Every constraint function is deterministic and operates in-place where possible.

Constraints enforced:
    - 0 < exposure <= 1
    - duration in (0, 366] (days)
    - insured_age in [16, 90]
    - credit_score in [400, 900]
    - years_no_claims >= 0
    - car_age in [-2, 20]
    - annual_miles_drive in [500, 60000]
    - annual_pct_driven in [0, 1]
    - total_miles_driven >= 0
    - avg_days_week in [0, 7]
    - All pct_drive_* in [0, 1]
    - sum(pct_drive_mon..sun) == 1 (compositional)
    - pct_drive_wkday == sum(mon..fri)
    - pct_drive_wkend == sum(sat, sun)
    - accel_6 >= accel_8 >= accel_9 >= accel_11 >= accel_12 >= accel_14 >= 0
    - brake_6 >= brake_8 >= brake_9 >= brake_11 >= brake_12 >= brake_14 >= 0
    - left/right turn intensities non-increasing in threshold (raw counts >= 0)
    - claim_count >= 0 (integer)
    - claim_amount >= 0
    - claim_amount == 0 when claim_count == 0
    - pure_premium == claim_amount / exposure (when exposure > 0)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from synthdrive.data.schema import (
    ACCEL_COLUMNS,
    BRAKE_COLUMNS,
    DOW_COLUMNS,
    LEFT_TURN_COLUMNS,
    RIGHT_TURN_COLUMNS,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ConstraintResult = Tuple[str, bool, str]  # (name, passed, message)


# ---------------------------------------------------------------------------
# Individual enforcement functions
# ---------------------------------------------------------------------------


def enforce_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Clip exposure to (0, 1]. Duration (in days) is not clipped here."""
    df = df.copy()
    if "exposure" in df.columns:
        df["exposure"] = df["exposure"].clip(lower=1e-6, upper=1.0)
    return df


def enforce_driver_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip driver variables to valid ranges."""
    df = df.copy()
    if "insured_age" in df.columns:
        df["insured_age"] = df["insured_age"].clip(lower=16, upper=90).astype("int64")
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].clip(lower=400, upper=900).astype("int64")
    if "years_no_claims" in df.columns:
        df["years_no_claims"] = df["years_no_claims"].clip(lower=0).astype("int64")
    return df


def enforce_vehicle_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip vehicle variables to valid ranges."""
    df = df.copy()
    if "car_age" in df.columns:
        df["car_age"] = df["car_age"].clip(lower=-2, upper=20).astype("int64")
    if "annual_miles_drive" in df.columns:
        df["annual_miles_drive"] = df["annual_miles_drive"].clip(lower=500.0, upper=60_000.0)
    return df


def enforce_telematics_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip individual telematics proportion and rate variables."""
    df = df.copy()
    if "annual_pct_driven" in df.columns:
        df["annual_pct_driven"] = df["annual_pct_driven"].clip(lower=0.0, upper=1.0)
    if "total_miles_driven" in df.columns:
        df["total_miles_driven"] = df["total_miles_driven"].clip(lower=0.0)
    if "avg_days_week" in df.columns:
        df["avg_days_week"] = df["avg_days_week"].clip(lower=0.0, upper=7.0)

    # Proportion columns in [0, 1]
    prop_cols = [
        "pct_drive_mon", "pct_drive_tue", "pct_drive_wed", "pct_drive_thu",
        "pct_drive_fri", "pct_drive_sat", "pct_drive_sun",
        "pct_drive_wkday", "pct_drive_wkend",
        "pct_drive_rush_am", "pct_drive_rush_pm",
        "pct_drive_2hrs", "pct_drive_3hrs", "pct_drive_4hrs",
    ]

    for col in prop_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0, upper=1.0)

    # Accel/brake event counts: non-negative integers
    for col in list(ACCEL_COLUMNS) + list(BRAKE_COLUMNS):
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0).round().astype("int64")

    # Turn intensity counts: non-negative integers (not proportions)
    for col in list(LEFT_TURN_COLUMNS) + list(RIGHT_TURN_COLUMNS):
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0).round().astype("int64")

    return df


def enforce_compositional_dow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize day-of-week proportions so they sum to 1.0, then recompute
    the derived weekday/weekend aggregates.

    If all seven DOW columns are present and their row sums are zero, the
    column is set to a uniform distribution (1/7 each) to avoid division by
    zero.
    """
    df = df.copy()
    present = [c for c in DOW_COLUMNS if c in df.columns]
    if len(present) < 7:
        # Cannot normalize if any component is missing; skip silently.
        return df

    dow_arr = df[list(DOW_COLUMNS)].values.astype(float)
    row_sums = dow_arr.sum(axis=1, keepdims=True)

    # Replace zero rows with uniform distribution
    zero_rows = (row_sums == 0).ravel()
    if zero_rows.any():
        dow_arr[zero_rows] = 1.0 / 7.0
        row_sums[zero_rows] = 1.0

    dow_normalized = dow_arr / row_sums
    df[list(DOW_COLUMNS)] = dow_normalized

    # Recompute derived aggregates
    df["pct_drive_wkday"] = df[
        ["pct_drive_mon", "pct_drive_tue", "pct_drive_wed",
         "pct_drive_thu", "pct_drive_fri"]
    ].sum(axis=1)
    df["pct_drive_wkend"] = df[["pct_drive_sat", "pct_drive_sun"]].sum(axis=1)

    return df


def enforce_monotone_thresholds(
    df: pd.DataFrame,
    columns: tuple,
) -> pd.DataFrame:
    """
    Enforce that the given columns are non-increasing from left to right
    within each row (i.e., columns[0] >= columns[1] >= ... >= columns[-1]).

    Used for accel_*_miles, brake_*_miles, and turn intensity sequences.
    Works by taking a running cumulative minimum across the column sequence.
    """
    df = df.copy()
    present = [c for c in columns if c in df.columns]
    if len(present) < 2:
        return df

    arr = df[present].values.astype(float)
    # Cumulative minimum across columns (left to right)
    arr = np.minimum.accumulate(arr, axis=1)
    df[present] = arr
    return df


def enforce_claim_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce claim consistency:
        - claim_count must be a non-negative integer
        - claim_amount must be >= 0
        - claim_amount == 0 whenever claim_count == 0
        - claim_count == 0 whenever claim_amount == 0 is not enforced in the
          other direction (a policy can have claims with zero payout)
    """
    df = df.copy()
    if "claim_count" in df.columns:
        df["claim_count"] = df["claim_count"].clip(lower=0).round().astype("int64")
    if "claim_amount" in df.columns:
        df["claim_amount"] = df["claim_amount"].clip(lower=0.0)
    if "claim_count" in df.columns and "claim_amount" in df.columns:
        zero_mask = df["claim_count"] == 0
        df.loc[zero_mask, "claim_amount"] = 0.0
    return df


def recompute_pure_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute pure_premium = claim_amount / exposure.

    If exposure is zero or missing, pure_premium is set to NaN.
    """
    df = df.copy()
    if "claim_amount" in df.columns and "exposure" in df.columns:
        exposure = df["exposure"].replace(0.0, np.nan)
        df["pure_premium"] = df["claim_amount"] / exposure
        df["pure_premium"] = df["pure_premium"].fillna(0.0).clip(lower=0.0)
    return df


# ---------------------------------------------------------------------------
# Master enforcement function
# ---------------------------------------------------------------------------


def enforce_all_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all constraints in the correct order and return a clean DataFrame.

    Call order matters:
        1. Clip individual variable bounds first.
        2. Normalize compositional variables.
        3. Enforce monotone threshold sequences.
        4. Enforce claim consistency.
        5. Recompute pure premium.
    """
    df = enforce_exposure(df)
    df = enforce_driver_bounds(df)
    df = enforce_vehicle_bounds(df)
    df = enforce_telematics_bounds(df)
    df = enforce_compositional_dow(df)
    df = enforce_monotone_thresholds(df, ACCEL_COLUMNS)
    df = enforce_monotone_thresholds(df, BRAKE_COLUMNS)
    df = enforce_monotone_thresholds(df, LEFT_TURN_COLUMNS)
    df = enforce_monotone_thresholds(df, RIGHT_TURN_COLUMNS)
    df = enforce_claim_consistency(df)
    df = recompute_pure_premium(df)
    return df


# ---------------------------------------------------------------------------
# Constraint checking (non-mutating audit)
# ---------------------------------------------------------------------------


def check_constraints(df: pd.DataFrame) -> List[ConstraintResult]:
    """
    Audit a DataFrame for constraint violations.

    Returns a list of (check_name, passed, message) tuples.
    Does not modify the DataFrame.
    """
    results: List[ConstraintResult] = []
    n = len(df)

    def _check(name: str, mask: pd.Series, detail: str = "") -> None:
        n_fail = int((~mask).sum())
        passed = n_fail == 0
        msg = "OK" if passed else f"{n_fail}/{n} rows fail. {detail}"
        results.append((name, passed, msg))

    def _col(name: str) -> bool:
        return name in df.columns

    # --- Exposure
    if _col("exposure"):
        _check("exposure > 0", df["exposure"] > 0, "exposure must be strictly positive")
        _check("exposure <= 1", df["exposure"] <= 1.0, "exposure must not exceed 1.0")
    if _col("duration"):
        _check("duration > 0", df["duration"] > 0)
        _check("duration <= 366", df["duration"] <= 366.0)

    # --- Driver
    if _col("insured_age"):
        _check("insured_age >= 16", df["insured_age"] >= 16)
        _check("insured_age <= 90", df["insured_age"] <= 90)
    if _col("credit_score"):
        _check("credit_score >= 400", df["credit_score"] >= 400)
        _check("credit_score <= 900", df["credit_score"] <= 900)
    if _col("years_no_claims"):
        _check("years_no_claims >= 0", df["years_no_claims"] >= 0)

    # --- Vehicle
    if _col("car_age"):
        _check("car_age >= -2", df["car_age"] >= -2)
        _check("car_age <= 20", df["car_age"] <= 20)
    if _col("annual_miles_drive"):
        _check("annual_miles_drive >= 500", df["annual_miles_drive"] >= 500)
        _check("annual_miles_drive <= 60000", df["annual_miles_drive"] <= 60_000)

    # --- Telematics proportions
    if _col("annual_pct_driven"):
        _check("annual_pct_driven in [0,1]",
               (df["annual_pct_driven"] >= 0) & (df["annual_pct_driven"] <= 1))
    if _col("total_miles_driven"):
        _check("total_miles_driven >= 0", df["total_miles_driven"] >= 0)
    if _col("avg_days_week"):
        _check("avg_days_week in [0,7]",
               (df["avg_days_week"] >= 0) & (df["avg_days_week"] <= 7))

    # --- Compositional DOW
    dow_present = [c for c in DOW_COLUMNS if _col(c)]
    if len(dow_present) == 7:
        dow_sums = df[list(DOW_COLUMNS)].sum(axis=1)
        _check(
            "sum(pct_drive_dow) == 1",
            np.abs(dow_sums - 1.0) < 1e-6,
            "day-of-week proportions must sum to 1.0",
        )
    if _col("pct_drive_wkday") and _col("pct_drive_wkend"):
        _check(
            "wkday + wkend == 1",
            np.abs(df["pct_drive_wkday"] + df["pct_drive_wkend"] - 1.0) < 1e-6,
        )

    # --- Monotone threshold sequences
    for seqname, cols in [
        ("accel sequence", ACCEL_COLUMNS),
        ("brake sequence", BRAKE_COLUMNS),
        ("left_turn sequence", LEFT_TURN_COLUMNS),
        ("right_turn sequence", RIGHT_TURN_COLUMNS),
    ]:
        present = [c for c in cols if _col(c)]
        for i in range(len(present) - 1):
            col_a, col_b = present[i], present[i + 1]
            _check(
                f"{col_a} >= {col_b}",
                df[col_a] >= df[col_b] - 1e-9,
                f"threshold sequence {seqname} not non-increasing",
            )

    # --- Claims
    if _col("claim_count"):
        _check("claim_count >= 0", df["claim_count"] >= 0)
        _check("claim_count is integer",
               (df["claim_count"] == df["claim_count"].round()).all()
               * pd.Series([True] * n),
               "claim_count must be a non-negative integer")
    if _col("claim_amount"):
        _check("claim_amount >= 0", df["claim_amount"] >= 0)
    if _col("claim_count") and _col("claim_amount"):
        zero_mask = df["claim_count"] == 0
        _check(
            "claim_amount == 0 when claim_count == 0",
            ~zero_mask | (df["claim_amount"] == 0.0),
            "claim_amount must be 0 when claim_count is 0",
        )
    if _col("pure_premium"):
        _check("pure_premium >= 0", df["pure_premium"] >= 0)

    return results


def all_passed(results: List[ConstraintResult]) -> bool:
    """Return True if every constraint check in results passed."""
    return all(r[1] for r in results)


def format_constraint_report(results: List[ConstraintResult]) -> str:
    """Return a human-readable constraint audit report."""
    lines = ["Constraint Audit", "=" * 50]
    n_pass = sum(1 for r in results if r[1])
    n_fail = len(results) - n_pass
    lines.append(f"Passed: {n_pass}  |  Failed: {n_fail}  |  Total: {len(results)}")
    lines.append("")

    if n_fail > 0:
        lines.append("FAILURES:")
        for name, passed, msg in results:
            if not passed:
                lines.append(f"  FAIL  {name}: {msg}")
        lines.append("")

    lines.append("PASSES:")
    for name, passed, msg in results:
        if passed:
            lines.append(f"  pass  {name}")

    return "\n".join(lines)
