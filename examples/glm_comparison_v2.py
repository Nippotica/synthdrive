"""
examples/glm_comparison_v2.py
-------------------------------
GLM relativity comparison (v2): SBV seed vs SynthDrive-generated data.

Uses a Poisson GLM specification that matches the SynthDrive frequency model
exactly: binary age-band indicators, centered continuous predictors, and
log1p-transformed brake/accel counts.  Contrast with glm_comparison.py, which
uses quantile-banded categorical factors.

Centering constants (from SynthDrive formal spec v2):
  log_miles centred at log(12 000)
  pct_driven centred at 0.33
  car_age centred at 9
  log(1 + brake_9) centred at log(1 + 2) = log(3)
  log(1 + accel_9) centred at log(1 + 0) = 0

Output
------
  output/glm_comparison_v2.csv  — coefficient comparison table

Usage
-----
  python examples/glm_comparison_v2.py
  python examples/glm_comparison_v2.py \\
      --n 100000 --output output/ --random-state 42
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthdrive import generate

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Column renaming for SBV dot-notation CSV
# ---------------------------------------------------------------------------
_SBV_RENAME: dict[str, str] = {
    "Duration":           "duration",
    "Insured.age":        "insured_age",
    "Insured.sex":        "insured_sex",
    "Car.age":            "car_age",
    "Marital":            "marital_status",
    "Car.use":            "car_use",
    "Credit.score":       "credit_score",
    "Region":             "region",
    "Annual.miles.drive": "annual_miles_drive",
    "Years.noclaims":     "years_no_claims",
    "Territory":          "territory",
    "Annual.pct.driven":  "annual_pct_driven",
    "Total.miles.driven": "total_miles_driven",
    "Avgdays.week":       "avg_days_week",
    "NB_Claim":           "claim_count",
    "AMT_Claim":          "claim_amount",
    "NB_Freq":            "claim_count",
    "AMT_Claims":         "claim_amount",
    "Exposure":           "exposure",
    "Accel.09miles":      "accel_9_miles",
    "Brake.09miles":      "brake_9_miles",
}

# ---------------------------------------------------------------------------
# Centering constants — must match SynthDrive formal spec
# ---------------------------------------------------------------------------
_LOG_12000 = math.log(12_000)   # log annual miles reference
_LOG_3     = math.log(3)        # log(1 + tilde_b_9) = log(1 + 2)
_LOG_1     = 0.0                # log(1 + tilde_a_9) = log(1 + 0)


# ---------------------------------------------------------------------------
# GLM formula — all predictors are pre-transformed
# ---------------------------------------------------------------------------
_GLM_FORMULA = (
    "claim_count"
    " ~ young + senior"
    " + male + single"
    " + credit_c"
    " + commercial + farmer"
    " + log_miles_c + pct_c + cage_c"
    " + log_brake_c + log_accel_c"
)

_REQUIRED_COLS = {
    "claim_count", "log_exposure",
    "young", "senior",
    "male", "single",
    "credit_c",
    "commercial", "farmer",
    "log_miles_c", "pct_c", "cage_c",
    "log_brake_c", "log_accel_c",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sbv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    rename = {k: v for k, v in _SBV_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)

    if "exposure" not in df.columns:
        if "duration" in df.columns:
            df["exposure"] = df["duration"] / 365.0
        else:
            raise KeyError("Cannot derive exposure: neither 'Exposure' nor 'Duration' found.")

    df["exposure"] = df["exposure"].clip(lower=1e-6, upper=1.0)

    if "claim_count" in df.columns:
        df["claim_count"] = (
            df["claim_count"].fillna(0).astype(float).astype(int).clip(lower=0)
        )

    for col in ("insured_sex", "marital_status", "car_use"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# ---------------------------------------------------------------------------
# Predictor construction — matches SynthDrive risk score f(X_i) exactly
# ---------------------------------------------------------------------------
def add_model_predictors(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Create the transformed predictor columns that map 1-to-1 to the
    SynthDrive log-relative-risk score terms.
    """
    df = df.copy()

    # Age binary indicators
    df["young"]  = (df["insured_age"] < 25).astype(float)
    df["senior"] = (df["insured_age"] > 65).astype(float)

    # Sex and marital status
    df["male"]   = (df["insured_sex"].str.strip() == "Male").astype(float)
    df["single"] = (df["marital_status"].str.strip() == "Single").astype(float)

    # Credit score: centered at 650, scaled per 100 pts
    df["credit_c"] = (pd.to_numeric(df["credit_score"], errors="coerce") - 650.0) / 100.0

    # Car-use binary indicators (reference = Commute + Private)
    df["commercial"] = (df["car_use"].str.strip() == "Commercial").astype(float)
    df["farmer"]     = (df["car_use"].str.strip() == "Farmer").astype(float)

    # Log annual miles centered at log(12 000)
    miles = pd.to_numeric(df["annual_miles_drive"], errors="coerce").clip(lower=1.0)
    df["log_miles_c"] = np.log(miles) - _LOG_12000

    # Annual pct driven centered at 0.33
    df["pct_c"] = pd.to_numeric(df["annual_pct_driven"], errors="coerce") - 0.33

    # Car age centered at 9
    df["cage_c"] = pd.to_numeric(df["car_age"], errors="coerce") - 9.0

    # log(1 + brake_9) centered at log(3)
    if "brake_9_miles" in df.columns:
        brake = pd.to_numeric(df["brake_9_miles"], errors="coerce").clip(lower=0)
        df["log_brake_c"] = np.log1p(brake) - _LOG_3
    else:
        print(f"  [{label}] Warning: 'brake_9_miles' not found; log_brake_c = NaN.")
        df["log_brake_c"] = np.nan

    # log(1 + accel_9) centered at log(1) = 0
    if "accel_9_miles" in df.columns:
        accel = pd.to_numeric(df["accel_9_miles"], errors="coerce").clip(lower=0)
        df["log_accel_c"] = np.log1p(accel) - _LOG_1
    else:
        print(f"  [{label}] Warning: 'accel_9_miles' not found; log_accel_c = NaN.")
        df["log_accel_c"] = np.nan

    # Log exposure offset
    df["log_exposure"] = np.log(
        pd.to_numeric(df["exposure"], errors="coerce").clip(lower=1e-9)
    )

    return df


# ---------------------------------------------------------------------------
# GLM fitting
# ---------------------------------------------------------------------------
def fit_poisson_glm(df: pd.DataFrame, label: str) -> pd.Series:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{label}] Missing columns: {sorted(missing)}")

    cols = list(_REQUIRED_COLS)
    sub = df[cols].dropna()

    if len(sub) < len(df):
        n_dropped = len(df) - len(sub)
        print(f"  [{label}] Dropped {n_dropped:,} rows with NaN in predictors.")

    result = smf.glm(
        formula=_GLM_FORMULA,
        data=sub,
        family=sm.families.Poisson(),
        offset=sub["log_exposure"],
    ).fit(disp=False)

    return result.params


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def build_comparison_table(
    params_sbv: pd.Series,
    params_syn: pd.Series,
) -> pd.DataFrame:
    all_terms = sorted(set(params_sbv.index) | set(params_syn.index))
    rows = []
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
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{: .4f}".format)
    pd.set_option("display.width", 120)

    print("\n" + "=" * 72)
    print("  GLM v2: SBV vs SynthDrive — model-matched Poisson specification")
    print("  (binary age indicators, centered continuous terms, log1p brake/accel)")
    print("=" * 72)
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
            print(f"    {term:<35s}  {val:+.1f}%")
    else:
        print("\n  All terms within ±25% of SBV values.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLM comparison v2: model-matched Poisson specification."
    )
    parser.add_argument(
        "--seed", default="data/raw/telematics_syn-032021.csv",
        help="Path to the SBV seed CSV.",
    )
    parser.add_argument(
        "--n", type=int, default=100_000,
        help="Number of synthetic policies (default: 100000).",
    )
    parser.add_argument(
        "--output", default="output/",
        help="Output directory (default: output/).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for SynthDrive generation (default: 42).",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Load SBV ---
    print(f"Loading SBV seed data from: {args.seed}")
    sbv_raw = load_sbv(args.seed)
    print(f"  Rows: {len(sbv_raw):,}   Columns: {len(sbv_raw.columns)}")

    # --- Generate SynthDrive ---
    print(f"\nGenerating SynthDrive dataset (n={args.n:,}, seed mode) ...")
    syn_raw = generate(
        n=args.n,
        preset="core",
        seed_path=args.seed,
        random_state=args.random_state,
    )
    print(f"  Rows: {len(syn_raw):,}   Columns: {len(syn_raw.columns)}")

    # --- Build predictors ---
    print("\nConstructing model-matched predictors ...")
    sbv = add_model_predictors(sbv_raw, label="SBV")
    syn = add_model_predictors(syn_raw, label="SynthDrive")

    # --- Fit GLMs ---
    print("\nFitting Poisson GLM on SBV data ...")
    params_sbv = fit_poisson_glm(sbv, label="SBV")
    print(f"  Converged.  {len(params_sbv)} terms.")

    print("Fitting Poisson GLM on SynthDrive data ...")
    params_syn = fit_poisson_glm(syn, label="SynthDrive")
    print(f"  Converged.  {len(params_syn)} terms.")

    # --- Compare and save ---
    print("\nBuilding comparison table ...")
    table = build_comparison_table(params_sbv, params_syn)

    out_path = os.path.join(args.output, "glm_comparison_v2.csv")
    table.to_csv(out_path)
    print(f"Saved: {out_path}")

    print_summary(table)


if __name__ == "__main__":
    main()
