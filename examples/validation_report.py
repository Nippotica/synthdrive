"""
examples/validation_report.py
------------------------------
Generate a validation report comparing a SynthDrive-generated dataset
against the SBV seed dataset.

Produces figures in output/figures/ and a summary text in output/.

Required figures
----------------
  1. freq_by_age_band.png           Claim frequency by insured age band
  2. freq_by_territory.png          Claim frequency by territory quartile
  3. severity_histogram_log.png     Severity distribution on log scale
  4. feature_histograms.png         Grid of key feature marginal distributions
  5. correlation_heatmap.png        Spearman rank correlation heatmap
  6. dist_comparison_age.png        insured_age: SBV vs SynthDrive
  7. dist_comparison_miles.png      annual_miles_drive: SBV vs SynthDrive
  8. dist_comparison_claim_amount.png  claim_amount: SBV vs SynthDrive (log)

Usage
-----
  python examples/validation_report.py
  python examples/validation_report.py \\
      --n 100000 \\
      --seed data/raw/telematics_syn-032021.csv \\
      --output output/
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend; must precede pyplot import.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthdrive import generate

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Shared plot style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": "#cccccc",
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "image.cmap": "Greys",
    }
)

_SBV_COLOR = "#aaaaaa"   # Light gray — SBV seed data  (dashed)
_SYN_COLOR = "#333333"   # Dark gray  — SynthDrive      (solid)
_SBV_LS    = "--"
_SYN_LS    = "-"


# ---------------------------------------------------------------------------
# SBV loading (mirrors glm_comparison.py)
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
    "NB_Freq":             "claim_count",
    "AMT_Claims":          "claim_amount",
    "Exposure":            "exposure",
}


def load_sbv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    rename = {k: v for k, v in _SBV_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)
    if "exposure" not in df.columns and "duration" in df.columns:
        df["exposure"] = df["duration"] / 365.0
    df["exposure"] = df["exposure"].clip(lower=1e-6, upper=1.0)
    if "claim_count" in df.columns:
        df["claim_count"] = (
            df["claim_count"].fillna(0).astype(float).astype(int).clip(lower=0)
        )
    if "claim_amount" in df.columns:
        df["claim_amount"] = df["claim_amount"].fillna(0.0).astype(float).clip(lower=0.0)
    return df


def generate_synthdrive(
    n: int,
    seed_csv: str | None,
    random_state: int,
) -> pd.DataFrame:
    return generate(
        n=n,
        preset="core",
        seed_path=seed_csv,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Helper: compute exposure-weighted claim frequency by a grouping variable
# ---------------------------------------------------------------------------
def freq_by_group(
    df: pd.DataFrame,
    group_col: str,
) -> pd.Series:
    """
    Return claim frequency (claims per unit exposure) for each level of group_col.
    """
    grouped = df.groupby(group_col, observed=True)
    claims = grouped["claim_count"].sum()
    exposure = grouped["exposure"].sum()
    return (claims / exposure.replace(0, np.nan)).rename("frequency")


# ---------------------------------------------------------------------------
# Figure 1: Frequency by insured age band
# ---------------------------------------------------------------------------
def fig_freq_by_age(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    band_labels = ["16–25", "26–35", "36–45", "46–55", "56–65", "66–75", "76+"]
    x = np.arange(len(band_labels))
    width = 0.35

    for i, (df, label, color) in enumerate((
        (sbv, "SBV seed", _SBV_COLOR),
        (syn, "SynthDrive", _SYN_COLOR),
    )):
        banded = df.copy()
        banded["age_band"] = pd.cut(
            banded["insured_age"],
            bins=[15, 25, 35, 45, 55, 65, 75, 110],
            labels=band_labels,
        ).astype(str)
        freq = freq_by_group(banded, "age_band").reindex(band_labels)
        offset = (i - 0.5) * width
        ax.bar(x + offset, freq.values, width, label=label, color=color)

    ax.set_xlabel("Age Band")
    ax.set_ylabel("Claim Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.legend()
    ax.set_ylim(0, 0.10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    path = os.path.join(out_dir, "freq_by_age_band.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Frequency by territory (quartile band)
# ---------------------------------------------------------------------------
def fig_freq_by_territory(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    for df, label, color in (
        (sbv, "SBV seed", _SBV_COLOR),
        (syn, "SynthDrive", _SYN_COLOR),
    ):
        banded = df.copy()
        banded["terr_band"] = pd.qcut(
            banded["territory"],
            q=4,
            labels=["T-Q1 (low)", "T-Q2", "T-Q3", "T-Q4 (high)"],
            duplicates="drop",
        ).astype(str)
        freq = freq_by_group(banded, "terr_band")
        ax.bar(
            np.arange(len(freq)) + (0 if label == "SBV seed" else 0.35),
            freq.values,
            width=0.35,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(np.arange(4) + 0.175)
    ax.set_xticklabels(["T-Q1 (low)", "T-Q2", "T-Q3", "T-Q4 (high)"])
    ax.set_xlabel("Territory Quartile")
    ax.set_ylabel("Claim Frequency")
    ax.set_ylim(0, 0.06)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "freq_by_territory.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Severity histogram on log scale
# ---------------------------------------------------------------------------
def fig_severity_log(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, df, label, color in (
        (axes[0], sbv, "SBV seed", _SBV_COLOR),
        (axes[1], syn, "SynthDrive", _SYN_COLOR),
    ):
        positive = df.loc[df["claim_amount"] > 0, "claim_amount"]
        if len(positive) == 0:
            ax.text(0.5, 0.5, "No positive claims", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        log_vals = np.log10(positive)
        bins = np.linspace(0, 5, 35)
        ax.hist(log_vals, bins=bins, color=color, edgecolor="white", lw=0.4)
        ax.set_title(f"Severity Distribution — {label}", fontweight="bold")
        ax.set_xlabel("Claim Amount")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 5)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["$1", "$10", "$100", "$1,000", "$10,000", "$100,000"])

        # Overlay KDE
        ax2 = ax.twinx()
        kde = stats.gaussian_kde(log_vals)
        xgrid = np.linspace(0, 5, 200)
        ax2.plot(xgrid, kde(xgrid), color="black", lw=1.5, ls="--", label="KDE")
        ax2.set_ylabel("Density")
        ax2.spines["top"].set_visible(False)

        n_pos = len(positive)
        mean_sev = positive.mean()
        ax.text(
            0.97, 0.95,
            f"n = {n_pos:,}\nmean = ${mean_sev:,.0f}",
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    plt.suptitle("Claim Severity (log₁₀ scale, positive claims only)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "severity_histogram_log.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Feature histograms grid (SBV vs SynthDrive overlaid)
# ---------------------------------------------------------------------------
_FEATURE_GRID_COLS: list[tuple[str, str]] = [
    ("insured_age",        "Insured Age"),
    ("credit_score",       "Credit Score"),
    ("annual_miles_drive", "Annual Miles Drive"),
    ("annual_pct_driven",  "Annual Pct Driven"),
    ("car_age",            "Car Age"),
    ("years_no_claims",    "Years No Claims"),
    ("total_miles_driven", "Total Miles Driven"),
    ("avg_days_week",      "Avg Days / Week"),
]


def fig_feature_histograms(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    out_dir: str,
) -> None:
    # Keep only columns that exist in both datasets.
    cols_to_plot = [
        (col, label) for col, label in _FEATURE_GRID_COLS
        if col in sbv.columns and col in syn.columns
    ]

    n_cols = 4
    n_rows = int(np.ceil(len(cols_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, cols_to_plot):
        sbv_vals = pd.to_numeric(sbv[col], errors="coerce").dropna()
        syn_vals = pd.to_numeric(syn[col], errors="coerce").dropna()

        combined_min = min(sbv_vals.min(), syn_vals.min())
        combined_max = max(sbv_vals.max(), syn_vals.max())
        xgrid = np.linspace(combined_min, combined_max, 300)

        for vals, lbl, color, ls in (
            (sbv_vals, "SBV", _SBV_COLOR, _SBV_LS),
            (syn_vals, "SynthDrive", _SYN_COLOR, _SYN_LS),
        ):
            kde = stats.gaussian_kde(vals)
            ax.plot(xgrid, kde(xgrid), color=color, lw=1.5, ls=ls, label=lbl)

        ax.set_xlim(combined_min, combined_max)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    # Hide any unused axes.
    for ax in axes[len(cols_to_plot):]:
        ax.set_visible(False)

    plt.suptitle(
        "Feature Marginal Distributions: SBV vs SynthDrive",
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_histograms.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Spearman rank correlation heatmap (SynthDrive)
# ---------------------------------------------------------------------------
_CORR_COLS: list[str] = [
    "insured_age",
    "credit_score",
    "annual_miles_drive",
    "annual_pct_driven",
    "car_age",
    "years_no_claims",
    "total_miles_driven",
    "avg_days_week",
    "claim_count",
    "claim_amount",
]


def fig_correlation_heatmap(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, label in (
        (axes[0], sbv, "SBV seed"),
        (axes[1], syn, "SynthDrive"),
    ):
        cols = [c for c in _CORR_COLS if c in df.columns]
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        corr = numeric.corr(method="spearman")

        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="Greys", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(f"Spearman Rank Correlation — {label}", fontweight="bold")

        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="white" if val > 0.5 else "black",
                    )

        plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    plt.suptitle(
        "Spearman Rank Correlation Heatmaps", fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figures 6-8: Distribution comparisons (three individual variables)
# ---------------------------------------------------------------------------
def fig_distribution_comparison(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    col: str,
    title: str,
    out_dir: str,
    log_scale: bool = False,
) -> None:
    sbv_vals = pd.to_numeric(sbv[col], errors="coerce").dropna() if col in sbv.columns else pd.Series(dtype=float)
    syn_vals = pd.to_numeric(syn[col], errors="coerce").dropna() if col in syn.columns else pd.Series(dtype=float)

    if len(sbv_vals) == 0 or len(syn_vals) == 0:
        print(f"  Warning: no data for '{col}'; skipping distribution comparison.")
        return

    if log_scale:
        sbv_vals = sbv_vals[sbv_vals > 0]
        syn_vals = syn_vals[syn_vals > 0]
        sbv_plot = np.log10(sbv_vals)
        syn_plot = np.log10(syn_vals)
        xlabel = f"log₁₀({col})"
    else:
        sbv_plot = sbv_vals
        syn_plot = syn_vals
        xlabel = col

    combined_min = min(sbv_plot.min(), syn_plot.min())
    combined_max = max(sbv_plot.max(), syn_plot.max())
    bins = np.linspace(combined_min, combined_max, 40)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, vals, plot_vals, label, color in (
        (axes[0], sbv_vals, sbv_plot, "SBV seed", _SBV_COLOR),
        (axes[1], syn_vals, syn_plot, "SynthDrive", _SYN_COLOR),
    ):
        ax.hist(plot_vals, bins=bins, color=color,
                edgecolor="white", lw=0.4, density=True)

        kde = stats.gaussian_kde(plot_vals)
        xgrid = np.linspace(combined_min, combined_max, 300)
        ax.plot(xgrid, kde(xgrid), color="black", lw=1.5, ls="--", label="KDE")

        ax.set_title(f"{title} — {label}", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        summary = (
            f"n = {len(vals):,}\n"
            f"mean = {vals.mean():.2f}\n"
            f"median = {vals.median():.2f}\n"
            f"std = {vals.std():.2f}"
        )
        ax.text(
            0.97, 0.95, summary,
            ha="right", va="top", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    plt.suptitle(f"Distribution Comparison: {title}", fontweight="bold")
    plt.tight_layout()
    safe_name = col.replace("_", "-")
    path = os.path.join(out_dir, f"dist_comparison_{safe_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Text summary report
# ---------------------------------------------------------------------------
def write_text_summary(
    sbv: pd.DataFrame,
    syn: pd.DataFrame,
    out_dir: str,
) -> None:
    lines: list[str] = ["SynthDrive v0.1 — Validation Report", "=" * 50, ""]

    # Exposure
    lines += [
        "--- Exposure ---",
        f"  SBV:         mean={sbv['exposure'].mean():.4f}  "
        f"min={sbv['exposure'].min():.4f}  max={sbv['exposure'].max():.4f}",
        f"  SynthDrive:  mean={syn['exposure'].mean():.4f}  "
        f"min={syn['exposure'].min():.4f}  max={syn['exposure'].max():.4f}",
        "",
    ]

    # Claim count
    for label, df in (("SBV", sbv), ("SynthDrive", syn)):
        vc = df["claim_count"].value_counts().sort_index()
        total = len(df)
        pct_zero = 100.0 * vc.get(0, 0) / total
        freq = df["claim_count"].sum() / df["exposure"].sum()
        lines.append(
            f"--- Claim Count [{label}] ---"
        )
        for val, cnt in vc.items():
            lines.append(f"  NB_Claim={val}: {cnt:>7,}  ({100*cnt/total:.2f}%)")
        lines += [
            f"  Zero-claim %: {pct_zero:.2f}%",
            f"  Claim frequency (per unit exposure): {freq:.4f}",
            "",
        ]

    # Severity
    for label, df in (("SBV", sbv), ("SynthDrive", syn)):
        pos = df.loc[df["claim_amount"] > 0, "claim_amount"]
        if len(pos) > 0:
            lines.append(f"--- Severity [{label}] (positive claims only) ---")
            lines += [
                f"  n       = {len(pos):,}",
                f"  mean    = {pos.mean():,.2f}",
                f"  median  = {pos.median():,.2f}",
                f"  std     = {pos.std():,.2f}",
                f"  p95     = {pos.quantile(0.95):,.2f}",
                f"  max     = {pos.max():,.2f}",
                "",
            ]

    path = os.path.join(out_dir, "validation_summary.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SynthDrive validation report figures."
    )
    parser.add_argument(
        "--n", type=int, default=100_000,
        help="Number of synthetic policies (default: 100000).",
    )
    parser.add_argument(
        "--seed", default="data/raw/telematics_syn-032021.csv",
        help="Path to the SBV seed CSV.",
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

    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading SBV seed data from: {args.seed}")
    sbv = load_sbv(args.seed)
    print(f"  Rows: {len(sbv):,}")

    print(f"\nGenerating SynthDrive dataset (n={args.n:,}) ...")
    syn = generate_synthdrive(
        n=args.n,
        seed_csv=args.seed,
        random_state=args.random_state,
    )
    print(f"  Rows: {len(syn):,}")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print(f"\nGenerating figures in: {fig_dir}")

    print("\n[1/8] Frequency by age band")
    fig_freq_by_age(sbv, syn, fig_dir)

    print("[2/8] Frequency by territory")
    fig_freq_by_territory(sbv, syn, fig_dir)

    print("[3/8] Severity histogram (log scale)")
    fig_severity_log(sbv, syn, fig_dir)

    print("[4/8] Feature histograms grid")
    fig_feature_histograms(sbv, syn, fig_dir)

    print("[5/8] Correlation heatmap")
    fig_correlation_heatmap(sbv, syn, fig_dir)

    print("[6/8] Distribution comparison: insured_age")
    fig_distribution_comparison(
        sbv, syn, "insured_age", "Insured Age", fig_dir, log_scale=False
    )

    print("[7/8] Distribution comparison: annual_miles_drive")
    fig_distribution_comparison(
        sbv, syn, "annual_miles_drive", "Annual Miles Drive", fig_dir, log_scale=False
    )

    print("[8/8] Distribution comparison: claim_amount (log scale)")
    fig_distribution_comparison(
        sbv, syn, "claim_amount", "Claim Amount", fig_dir, log_scale=True
    )

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    print("\nWriting text summary ...")
    write_text_summary(sbv, syn, args.output)

    print("\nValidation report complete.")
    print(f"Figures: {fig_dir}")
    print(f"Summary: {os.path.join(args.output, 'validation_summary.txt')}")


if __name__ == "__main__":
    main()
