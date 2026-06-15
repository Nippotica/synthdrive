"""
Example: Generate a 100 000-policy synthetic telematics portfolio.

Usage
-----
    python examples/generate_core_dataset.py

Output
------
    synthdrive_portfolio_100k.csv  (~55 columns, 100 000 rows)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import synthdrive


def main() -> None:
    print("SynthDrive v0.1 — Core Dataset Generation Example")
    print("=" * 55)

    n = 100_000
    print(f"Generating {n:,} synthetic policies...")
    t0 = time.time()

    df = synthdrive.generate(n=n, preset="core", random_state=42)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({n / elapsed:,.0f} policies/sec)")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()

    # Quick summary
    total_exp = df["exposure"].sum()
    freq = df["claim_count"].sum() / total_exp
    zero_pct = (df["claim_count"] == 0).mean()
    claimants = df[df["claim_amount"] > 0]["claim_amount"]
    mean_sev = claimants.mean() if len(claimants) > 0 else 0.0

    print("Portfolio summary")
    print("-" * 30)
    print(f"  Total exposure:  {total_exp:,.0f} policy-years")
    print(f"  Claim frequency: {freq:.4f} per policy-year")
    print(f"  Zero-claim pct:  {zero_pct:.1%}")
    print(f"  Mean severity:   {mean_sev:,.0f} (claimants only)")
    print(f"  Mean pure prem:  {df['pure_premium'].mean():,.2f}")
    print()

    # Save
    out_path = Path("synthdrive_portfolio_100k.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1_048_576:.1f} MB)")

    # Preview
    print()
    print("First 5 rows (selected columns):")
    preview_cols = [
        "policy_id", "duration", "insured_age", "insured_sex",
        "credit_score", "annual_miles_drive", "claim_count",
        "claim_amount", "pure_premium",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
