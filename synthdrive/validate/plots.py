"""
Validation plots for SynthDrive v0.1.

All functions return matplotlib Figure objects.  They do not call plt.show()
so that they can be used in headless environments and embedded in reports.

Use the 'Agg' backend by default to avoid display requirements:
    import matplotlib
    matplotlib.use('Agg')
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use Agg backend for headless compatibility
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
_GRID_ALPHA = 0.3


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=_GRID_ALPHA)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Frequency relativities
# ---------------------------------------------------------------------------


def plot_frequency_by_variable(
    df: pd.DataFrame,
    variable: str,
    n_bands: int = 6,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """
    Bar chart of claim frequency by quantile band of a continuous variable.
    """
    from synthdrive.validate.diagnostics import frequency_by_band

    band_df = frequency_by_band(df, variable, n_bands=n_bands)
    if "frequency" not in band_df.columns or len(band_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    bands = band_df["band"].astype(str)
    freq = band_df["frequency"].values * 100.0  # convert to %

    bars = ax.bar(range(len(bands)), freq, color=_PALETTE[0], alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, rotation=30, ha="right", fontsize=7)

    # Annotate bars
    for bar, val in zip(bars, freq):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.05,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    _apply_style(
        ax,
        title=title or f"Claim Frequency by {variable}",
        xlabel=variable,
        ylabel="Frequency (% per policy-year)",
    )
    fig.tight_layout()
    return fig


def plot_frequency_by_age(df: pd.DataFrame, figsize=(7, 4)) -> plt.Figure:
    return plot_frequency_by_variable(
        df, "insured_age", n_bands=6,
        title="Claim Frequency by Insured Age Band",
        figsize=figsize,
    )


def plot_frequency_by_territory(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Bar chart of claim frequency by territory."""
    if "territory" not in df.columns or "claim_count" not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No territory or claim data", ha="center", va="center")
        return fig

    grp = (
        df.groupby("territory", observed=True)
        .agg(
            n_policies=("claim_count", "count"),
            claim_count=("claim_count", "sum"),
            exposure=("exposure", "sum") if "exposure" in df.columns else ("claim_count", "count"),
        )
        .reset_index()
    )
    safe_exp = grp["exposure"].replace(0.0, np.nan)
    grp["frequency"] = grp["claim_count"] / safe_exp * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    terr = grp["territory"].astype(str)
    bars = ax.bar(terr, grp["frequency"], color=_PALETTE[1], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, grp["frequency"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.05,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=7,
        )
    _apply_style(ax, "Claim Frequency by Territory", "Territory", "Frequency (%/yr)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Severity distribution
# ---------------------------------------------------------------------------


def plot_severity_histogram(
    df: pd.DataFrame,
    log_scale: bool = True,
    n_bins: int = 40,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Histogram of claim amounts for claimant policies."""
    if "claim_amount" not in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No claim data", ha="center", va="center")
        return fig

    sev = df.loc[df["claim_amount"] > 0, "claim_amount"].values
    if len(sev) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No claimants", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    data_plot = np.log10(sev) if log_scale else sev
    ax.hist(data_plot, bins=n_bins, color=_PALETTE[2], alpha=0.80, edgecolor="white")

    xlabel = "log₁₀(Claim Amount)" if log_scale else "Claim Amount"
    _apply_style(ax, "Claim Severity Distribution (claimants only)", xlabel, "Count")

    # Add mean/median lines
    mean_val = np.mean(data_plot)
    med_val = np.median(data_plot)
    ax.axvline(mean_val, color=_PALETTE[3], lw=1.5, linestyle="--", label=f"Mean")
    ax.axvline(med_val, color=_PALETTE[4], lw=1.5, linestyle=":", label=f"Median")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature distributions
# ---------------------------------------------------------------------------


def plot_feature_histograms(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Grid of histograms for specified continuous variables.
    """
    if variables is None:
        variables = [
            "insured_age", "credit_score", "annual_miles_drive",
            "avg_days_week", "annual_pct_driven", "total_miles_driven",
        ]
    variables = [v for v in variables if v in df.columns]
    if not variables:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
        return fig

    n_rows = int(np.ceil(len(variables) / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 2.8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, var in enumerate(variables):
        ax = axes_flat[i]
        data = df[var].dropna().values.astype(float)
        ax.hist(data, bins=30, color=_PALETTE[i % len(_PALETTE)], alpha=0.80, edgecolor="white")
        ax.set_title(var, fontsize=8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=_GRID_ALPHA)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for j in range(len(variables), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 7),
    method: str = "spearman",
) -> plt.Figure:
    """
    Spearman rank correlation heatmap for continuous variables.
    """
    if variables is None:
        variables = [
            "insured_age", "credit_score", "years_no_claims",
            "car_age", "annual_miles_drive", "avg_days_week",
            "annual_pct_driven", "accel_9_miles", "brake_9_miles",
            "claim_count",
        ]
    variables = [v for v in variables if v in df.columns]
    if len(variables) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient columns", ha="center", va="center")
        return fig

    corr = df[variables].corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(variables)))
    ax.set_yticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(variables, fontsize=7)

    # Annotate cells
    for i in range(len(variables)):
        for j in range(len(variables)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_title(
        f"{method.capitalize()} Rank Correlation Matrix",
        fontsize=11, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distribution comparison (synth vs reference)
# ---------------------------------------------------------------------------


def plot_distribution_comparison(
    synth: pd.DataFrame,
    reference: pd.DataFrame,
    variable: str,
    n_bins: int = 30,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """
    Overlaid histograms comparing a variable between synthetic and reference data.
    """
    if variable not in synth.columns or variable not in reference.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"'{variable}' not found", ha="center", va="center")
        return fig

    s_data = synth[variable].dropna().values.astype(float)
    r_data = reference[variable].dropna().values.astype(float)

    all_data = np.concatenate([s_data, r_data])
    bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(s_data, bins=bins, density=True, alpha=0.55,
            color=_PALETTE[0], label="Synthetic")
    ax.hist(r_data, bins=bins, density=True, alpha=0.55,
            color=_PALETTE[1], label="Reference")
    ax.legend(fontsize=8)
    _apply_style(ax, f"Distribution Comparison: {variable}", variable, "Density")
    fig.tight_layout()
    return fig
