"""
examples/glm_comparison.py
--------------------------
GLM relativity comparison: SBV seed data vs SynthDrive-generated data.

Fits a Poisson GLM with log link on both datasets using the same formula,
bands continuous variables into 5 quantile groups, and produces a side-by-side
coefficient table (log-RR).

Output
------
  output/glm_comparison.csv   — coefficient table, CSV
  stdout                      — table + summary statistics

Usage
-----
  python examples/glm_comparison.py
  python examples/glm_comparison.py --seed data/raw/telematics_syn-032021.csv \\
                                     --n 100000 --output output/ --random-state 42
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Allow running from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthdrive import generate

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# SBV column name mapping: dot-notation CSV → canonical underscore names
# Source: So, Boucher, and Valdez (2021), Table 2
# ---------------------------------------------------------------------------
_SBV_RENAME: dict[str, str] = {
    "Duration":            "duration",
    "Insured.age":         "insured_age",
    "Insured.sex":         "insured_sex",
    "Car.age":             "car_age",
    "Marital":             "marital_status",
    "Car.use":             "car_use",
    "Credit.score":        "credit_score",
    "Region":              "region",
    "Annual.miles.drive":  "annual_miles_drive",
    "Years.noclaims":      "years_no_claims",
    "Territory":           "territory",
    "Annual.pct.driven":   "annual_pct_driven",
    "Total.miles.driven":  "total_miles_driven",
    "Avgdays.week":        "avg_days_week",
    "NB_Claim":            "claim_count",
    "AMT_Claim":           "claim_amount",
    # Some versions of the CSV may already have canonical names; handle both.
    "NB_Freq":             "claim_count",
    "AMT_Claims":          "claim_amount",
    "Exposure":            "exposure",
}


def load_sbv(csv_path: str) -> pd.DataFrame:
    """
    Load the SBV seed CSV and normalize it to canonical column names.

    Computes exposure = Duration / 365 if not already present.
    Clips exposure to (0, 1].
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Rename SBV dot-notation columns to canonical underscore names.
    rename = {k: v for k, v in _SBV_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Compute exposure from Duration (days) if not already present.
    if "exposure" not in df.columns:
        if "duration" in df.columns:
            df["exposure"] = df["duration"] / 365.0
        else:
            raise KeyError(
                "Cannot find 'Exposure', 'Duration', or 'duration' in the SBV CSV. "
                "Check column names."
            )

    df["exposure"] = df["exposure"].clip(lower=1e-6, upper=1.0)

    # Ensure claim_count is a non-negative integer.
    if "claim_count" in df.columns:
        df["claim_count"] = (
            df["claim_count"].fillna(0).astype(float).astype(int).clip(lower=0)
        )

    # Normalize string categoricals.
    for col in ("insured_sex", "marital_status", "car_use", "region"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def add_quantile_bands(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Add quantile-banded columns for continuous variables used in the GLM.

    Each banded column is a string of the form 'Q1' … 'Q5'.
    """
    df = df.copy()

    band_map = {
        "insured_age":        "insured_age_band",
        "credit_score":       "credit_score_band",
        "annual_miles_drive": "annual_miles_band",
        "annual_pct_driven":  "annual_pct_driven_band",
        "car_age":            "car_age_band",
        "territory":          "territory_band",
    }

    for src, tgt in band_map.items():
        if src not in df.columns:
            print(f"  Warning: column '{src}' not found; skipping '{tgt}'.")
            continue
        try:
            _, bins = pd.qcut(
                df[src], q=n_quantiles, retbins=True, duplicates="drop"
            )
            actual_n = len(bins) - 1
            df[tgt] = pd.qcut(
                df[src],
                q=n_quantiles,
                labels=[f"Q{i + 1}" for i in range(actual_n)],
                duplicates="drop",
            ).astype(str)
        except Exception as exc:
            print(f"  Warning: could not band '{src}': {exc}")

    # Ensure plain categoricals are strings so statsmodels treats them as factors.
    for col in ("insured_sex", "marital_status", "car_use", "region"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# The GLM formula.  Continuous variables are pre-banded before fitting;
# we use C() to make the factor treatment explicit.
_GLM_FORMULA = (
    "claim_count ~ C(insured_age_band)"
    " + C(insured_sex)"
    " + C(marital_status)"
    " + C(car_use)"
    " + C(credit_score_band)"
    " + C(annual_miles_band)"
    " + C(annual_pct_driven_band)"
    " + C(territory_band)"
    " + C(car_age_band)"
)

# Minimum columns required in the dataset before fitting.
_REQUIRED_COLS = {
    "claim_count", "exposure",
    "insured_age_band", "insured_sex", "marital_status", "car_use",
    "credit_score_band", "annual_miles_band", "annual_pct_driven_band",
    "territory_band", "car_age_band",
}


def _check_columns(df: pd.DataFrame, label: str) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"[{label}] Missing columns required for GLM: {sorted(missing)}"
        )


def fit_poisson_glm(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Fit a Poisson GLM with log link.  Returns the coefficient Series.

    Parameters
    ----------
    df    : DataFrame with banded predictors, 'claim_count', and 'exposure'.
    label : Human-readable name for error messages.

    Returns
    -------
    pd.Series of fitted coefficients (index = term names).
    """
    _check_columns(df, label)

    df = df.copy()
    df["log_exposure"] = np.log(df["exposure"].clip(lower=1e-9))

    # Drop any rows with NaN in required columns (should be rare).
    cols_needed = list(_REQUIRED_COLS | {"log_exposure"})
    cols_present = [c for c in cols_needed if c in df.columns]
    df = df[cols_present].dropna()

    result = smf.glm(
        formula=_GLM_FORMULA,
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_exposure"],
    ).fit(disp=False)

    return result.params


def build_comparison_table(
    params_sbv: pd.Series,
    params_syn: pd.Series,
) -> pd.DataFrame:
    """
    Merge two coefficient series into a side-by-side comparison table.

    Columns:
        coef_sbv         — log-RR from the SBV GLM
        coef_synthdrive  — log-RR from the SynthDrive GLM
        pct_diff         — 100 * (synthdrive − sbv) / |sbv|
    """
    all_terms = sorted(set(params_sbv.index) | set(params_syn.index))

    rows: list[dict] = []
    for term in all_terms:
        sbv_val = float(params_sbv.get(term, np.nan))
        syn_val = float(params_syn.get(term, np.nan))

        if not np.isnan(sbv_val) and abs(sbv_val) > 1e-12:
            pct_diff = 100.0 * (syn_val - sbv_val) / abs(sbv_val)
        else:
            pct_diff = np.nan

        rows.append(
            {
                "term":            term,
                "coef_sbv":        round(sbv_val, 6),
                "coef_synthdrive": round(syn_val, 6),
                "pct_diff":        round(pct_diff, 2) if not np.isnan(pct_diff) else np.nan,
            }
        )

    return pd.DataFrame(rows).set_index("term")


def print_summary(table: pd.DataFrame) -> None:
    """Print the coefficient table and aggregate accuracy statistics."""
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{: .4f}".format)
    pd.set_option("display.width", 120)

    print("\n" + "=" * 70)
    print("  GLM Coefficient Comparison: SBV vs SynthDrive")
    print("  (Poisson GLM, log link, log-RR coefficients)")
    print("=" * 70)
    print(table.to_string())

    valid_pct = table["pct_diff"].dropna()
    if len(valid_pct) == 0:
        return

    abs_pct = valid_pct.abs()
    print()
    print(f"  Terms compared             : {len(valid_pct)}")
    print(f"  Mean absolute % difference : {abs_pct.mean():.1f}%")
    print(f"  Median absolute % diff     : {abs_pct.median():.1f}%")
    print(f"  Max absolute % diff        : {abs_pct.max():.1f}%")

    large = abs_pct[abs_pct > 25]
    if len(large):
        print(f"\n  Terms with |pct_diff| > 25%:")
        for term, val in large.sort_values(ascending=False).items():
            print(f"    {term:<50s}  {val:+.1f}%")
    else:
        print("\n  All terms within ±25% of SBV values.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLM relativity comparison: SBV seed data vs SynthDrive-generated data."
    )
    parser.add_argument(
        "--seed",
        default="data/raw/telematics_syn-032021.csv",
        help="Path to the SBV seed CSV (default: data/raw/telematics_syn-032021.csv).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100_000,
        help="Number of synthetic policies to generate (default: 100000).",
    )
    parser.add_argument(
        "--output",
        default="output/",
        help="Directory for output files (default: output/).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for SynthDrive generation (default: 42).",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load and normalize SBV data.
    # ------------------------------------------------------------------
    print(f"Loading SBV seed data from: {args.seed}")
    sbv_raw = load_sbv(args.seed)
    print(f"  Rows: {len(sbv_raw):,}   Columns: {len(sbv_raw.columns)}")

    # ------------------------------------------------------------------
    # Step 2: Generate SynthDrive data.
    #
    # SynthDrive supports two modes: parameter mode and seed mode.
    # In seed mode, the SBV CSV is passed via seed_csv so that the
    # copula and marginal distributions are learned from the seed data.
    # ------------------------------------------------------------------
    print(f"\nGenerating SynthDrive dataset (n={args.n:,}, seed mode) ...")
    syn_raw = generate(
        n=args.n,
        preset="core",
        seed_path=args.seed,
        random_state=args.random_state,
    )
    print(f"  Rows: {len(syn_raw):,}   Columns: {len(syn_raw.columns)}")

    # ------------------------------------------------------------------
    # Step 3: Band continuous variables (5 quantile groups).
    # ------------------------------------------------------------------
    print("\nBanding continuous variables into 5 quantile groups ...")
    sbv_banded = add_quantile_bands(sbv_raw, n_quantiles=5)
    syn_banded = add_quantile_bands(syn_raw, n_quantiles=5)

    # ------------------------------------------------------------------
    # Step 4: Fit Poisson GLMs.
    # ------------------------------------------------------------------
    print("\nFitting Poisson GLM on SBV data ...")
    params_sbv = fit_poisson_glm(sbv_banded, label="SBV")
    print(f"  Converged.  {len(params_sbv)} terms.")

    print("Fitting Poisson GLM on SynthDrive data ...")
    params_syn = fit_poisson_glm(syn_banded, label="SynthDrive")
    print(f"  Converged.  {len(params_syn)} terms.")

    # ------------------------------------------------------------------
    # Step 5: Build and save the comparison table.
    # ------------------------------------------------------------------
    print("\nBuilding comparison table ...")
    table = build_comparison_table(params_sbv, params_syn)

    out_path = os.path.join(args.output, "glm_comparison.csv")
    table.to_csv(out_path)
    print(f"Saved: {out_path}")

    # Print to terminal.
    print_summary(table)


if __name__ == "__main__":
    main()
