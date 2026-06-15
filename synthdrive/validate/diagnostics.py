"""
Validation diagnostics for SynthDrive v0.1.

Provides actuarially meaningful summary statistics and distribution comparisons.
All functions accept a DataFrame produced by synthdrive.generate() and return
plain Python dicts or DataFrames that are easy to inspect and serialize.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------


def portfolio_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute top-level portfolio statistics.

    Returns
    -------
    dict with keys:
        n_policies, total_exposure, mean_exposure,
        n_claims, zero_claim_pct, mean_frequency,
        mean_severity, median_severity, p75_severity, p95_severity,
        mean_pure_premium, median_pure_premium
    """
    stats: Dict[str, float] = {}
    stats["n_policies"] = len(df)

    if "exposure" in df.columns:
        stats["total_exposure"] = float(df["exposure"].sum())
        stats["mean_exposure"] = float(df["exposure"].mean())

    if "claim_count" in df.columns:
        n_zero = int((df["claim_count"] == 0).sum())
        n_claims = int((df["claim_count"] > 0).sum())
        stats["n_claimant_policies"] = n_claims
        stats["zero_claim_pct"] = n_zero / len(df)

        if "exposure" in df.columns and df["exposure"].sum() > 0:
            total_claims = int(df["claim_count"].sum())
            stats["claim_count_per_policy_year"] = (
                total_claims / float(df["exposure"].sum())
            )
        stats["mean_claim_count"] = float(df["claim_count"].mean())

    if "claim_amount" in df.columns:
        nonzero = df.loc[df["claim_amount"] > 0, "claim_amount"]
        if len(nonzero) > 0:
            stats["mean_severity"] = float(nonzero.mean())
            stats["median_severity"] = float(nonzero.median())
            stats["p75_severity"] = float(nonzero.quantile(0.75))
            stats["p95_severity"] = float(nonzero.quantile(0.95))
            stats["p99_severity"] = float(nonzero.quantile(0.99))
        else:
            for k in ("mean_severity", "median_severity", "p75_severity",
                      "p95_severity", "p99_severity"):
                stats[k] = 0.0

    if "pure_premium" in df.columns:
        stats["mean_pure_premium"] = float(df["pure_premium"].mean())
        stats["median_pure_premium"] = float(df["pure_premium"].median())
        nonzero_pp = df.loc[df["pure_premium"] > 0, "pure_premium"]
        if len(nonzero_pp) > 0:
            stats["mean_pure_premium_claimants"] = float(nonzero_pp.mean())

    return stats


# ---------------------------------------------------------------------------
# Frequency relativities by band
# ---------------------------------------------------------------------------


def frequency_by_band(
    df: pd.DataFrame,
    variable: str,
    n_bands: int = 5,
    use_quantile_bands: bool = True,
) -> pd.DataFrame:
    """
    Compute observed claim frequency by band of a given continuous variable.

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
        Column name.
    n_bands : int
        Number of bands.
    use_quantile_bands : bool
        If True, use quantile-based (equal-frequency) bands.
        If False, use equal-width bands.

    Returns
    -------
    pd.DataFrame with columns:
        band, n_policies, exposure, claim_count, frequency
    """
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in DataFrame.")

    data = df[[variable]].copy()
    if "claim_count" in df.columns:
        data["claim_count"] = df["claim_count"].values
    if "exposure" in df.columns:
        data["exposure"] = df["exposure"].values

    # Create bands
    if use_quantile_bands:
        quantiles = np.linspace(0, 1, n_bands + 1)
        bins = np.quantile(data[variable].dropna(), quantiles)
        bins = np.unique(bins)  # remove duplicates
    else:
        bins = np.linspace(data[variable].min(), data[variable].max(), n_bands + 1)

    labels = [
        f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
        for i in range(len(bins) - 1)
    ]
    data["band"] = pd.cut(
        data[variable],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    agg: Dict[str, str] = {"band": "first"}
    if "claim_count" in data.columns:
        agg["claim_count"] = "sum"
    if "exposure" in data.columns:
        agg["exposure"] = "sum"

    result = data.groupby("band", observed=True).agg(
        n_policies=("band", "count"),
        **{
            k: (k, v)
            for k, v in [
                ("claim_count", "sum"),
                ("exposure", "sum"),
            ]
            if k in data.columns
        },
    ).reset_index()

    if "exposure" in result.columns and "claim_count" in result.columns:
        safe_exp = result["exposure"].replace(0.0, np.nan)
        result["frequency"] = result["claim_count"] / safe_exp
    elif "claim_count" in result.columns:
        result["frequency"] = result["claim_count"] / result["n_policies"]

    return result


# ---------------------------------------------------------------------------
# Severity distribution summary
# ---------------------------------------------------------------------------


def severity_distribution(
    df: pd.DataFrame,
    percentiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Summarise the claim severity distribution for claimant policies.

    Returns a two-column DataFrame (statistic, value).
    """
    if "claim_amount" not in df.columns:
        return pd.DataFrame(columns=["statistic", "value"])

    if percentiles is None:
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    sev = df.loc[df["claim_amount"] > 0, "claim_amount"]
    rows = [
        ("n_claimants", len(sev)),
        ("mean", float(sev.mean()) if len(sev) > 0 else np.nan),
        ("std", float(sev.std()) if len(sev) > 0 else np.nan),
        ("cv", float(sev.std() / sev.mean()) if len(sev) > 0 else np.nan),
    ]
    for p in percentiles:
        rows.append((f"p{int(p * 100):02d}", float(sev.quantile(p)) if len(sev) > 0 else np.nan))

    return pd.DataFrame(rows, columns=["statistic", "value"])


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------


def compare_distributions(
    synth: pd.DataFrame,
    reference: pd.DataFrame,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare marginal distributions between synthetic and reference datasets.

    For each variable, reports the mean, std, and p50 for both datasets,
    plus percentage differences.

    Parameters
    ----------
    synth : pd.DataFrame
        Synthetic dataset (from generate()).
    reference : pd.DataFrame
        Reference dataset (e.g. loaded seed CSV).
    variables : list of str, optional
        Variables to compare.  If None, compares all numeric columns present
        in both DataFrames.

    Returns
    -------
    pd.DataFrame with columns:
        variable, synth_mean, ref_mean, mean_diff_pct,
        synth_std, ref_std, synth_p50, ref_p50
    """
    numeric_synth = synth.select_dtypes(include="number")
    numeric_ref = reference.select_dtypes(include="number")

    if variables is None:
        variables = sorted(set(numeric_synth.columns) & set(numeric_ref.columns))

    rows = []
    for var in variables:
        s = synth[var].dropna().values.astype(float)
        r = reference[var].dropna().values.astype(float)
        if len(s) == 0 or len(r) == 0:
            continue
        s_mean, r_mean = s.mean(), r.mean()
        mean_diff = (s_mean - r_mean) / (abs(r_mean) + 1e-10) * 100.0
        rows.append({
            "variable": var,
            "synth_mean": round(s_mean, 4),
            "ref_mean": round(r_mean, 4),
            "mean_diff_pct": round(mean_diff, 2),
            "synth_std": round(s.std(), 4),
            "ref_std": round(r.std(), 4),
            "synth_p50": round(np.median(s), 4),
            "ref_p50": round(np.median(r), 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quick sanity checks
# ---------------------------------------------------------------------------


def sanity_checks(df: pd.DataFrame) -> List[Tuple[str, bool, str]]:
    """
    Run a set of high-level sanity checks beyond the constraint audit.

    Returns list of (check_name, passed, message).
    """
    results: List[Tuple[str, bool, str]] = []

    def _add(name: str, condition: bool, ok_msg: str, fail_msg: str) -> None:
        results.append((name, condition, ok_msg if condition else fail_msg))

    n = len(df)

    # --- Claim frequency in plausible range (annualised, exposure-weighted)
    if "claim_count" in df.columns and "exposure" in df.columns:
        total_exp = df["exposure"].sum()
        if total_exp > 0:
            freq = df["claim_count"].sum() / total_exp
            _add(
                "claim_frequency_range",
                0.01 <= freq <= 0.50,
                f"OK — frequency = {freq:.4f}",
                f"Unusual — frequency = {freq:.4f} (expected 0.01–0.50)",
            )

    # --- Zero-claim pct in plausible range
    if "claim_count" in df.columns:
        zero_pct = (df["claim_count"] == 0).mean()
        _add(
            "zero_claim_pct_range",
            0.70 <= zero_pct <= 0.99,
            f"OK — zero-claim pct = {zero_pct:.3f}",
            f"Unusual — zero-claim pct = {zero_pct:.3f} (expected 0.70–0.99)",
        )

    # --- No negative amounts
    if "claim_amount" in df.columns:
        n_neg = int((df["claim_amount"] < 0).sum())
        _add("no_negative_amounts", n_neg == 0,
             "OK — no negative claim amounts",
             f"{n_neg} policies have negative claim amounts")

    # --- Proportion columns in [0, 1]
    prop_cols = [c for c in df.columns if c.startswith("pct_")]
    for col in prop_cols[:5]:  # spot-check first 5
        if col in df.columns:
            bad = int(((df[col] < 0) | (df[col] > 1)).sum())
            _add(f"{col}_in_01", bad == 0,
                 f"OK — {col} in [0,1]",
                 f"{bad}/{n} rows have {col} outside [0,1]")

    # --- Severity > min threshold for claimants
    if "claim_amount" in df.columns and "claim_count" in df.columns:
        claimants = df[df["claim_count"] > 0]
        if len(claimants) > 0:
            n_low = int((claimants["claim_amount"] < 10.0).sum())
            _add(
                "severity_above_minimum",
                n_low == 0,
                "OK — all claimant amounts > 10",
                f"{n_low} claimant policies have amount < 10",
            )

    # --- Age distribution sanity
    if "insured_age" in df.columns:
        mean_age = df["insured_age"].mean()
        _add("mean_age_range", 30 <= mean_age <= 60,
             f"OK — mean age = {mean_age:.1f}",
             f"Unusual — mean age = {mean_age:.1f} (expected 30–60)")

    return results
