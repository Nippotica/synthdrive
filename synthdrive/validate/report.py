"""
Validation report for SynthDrive v0.1.

Public API
----------
    report = synthdrive.validate(df)
    print(report.summary())
    report.save("output/")

    # Compare against reference seed data
    from synthdrive.data.load_seed import load_sbv_csv
    seed = load_sbv_csv("/data/sbv.csv")
    report = synthdrive.validate(df, reference_data=seed.df)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from synthdrive.features.constraints import (
    ConstraintResult,
    all_passed,
    check_constraints,
    format_constraint_report,
)
from synthdrive.validate.diagnostics import (
    compare_distributions,
    portfolio_summary,
    sanity_checks,
    severity_distribution,
)
from synthdrive.validate.plots import (
    plot_correlation_heatmap,
    plot_distribution_comparison,
    plot_feature_histograms,
    plot_frequency_by_age,
    plot_frequency_by_territory,
    plot_severity_histogram,
)


# ---------------------------------------------------------------------------
# ValidationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """
    Container for all validation results.

    Attributes
    ----------
    n_policies : int
        Number of rows in the validated DataFrame.
    passed_constraints : bool
        True if all constraint checks passed.
    constraint_results : list
        List of (check_name, passed, message) tuples.
    statistics : dict
        Portfolio-level summary statistics.
    sanity_check_results : list
        High-level sanity checks (list of (name, passed, message)).
    severity_summary : pd.DataFrame
        Severity distribution summary.
    figures : dict
        Generated matplotlib figures, keyed by name.
    comparison : pd.DataFrame or None
        Distribution comparison table vs reference data (if provided).
    """

    n_policies: int
    passed_constraints: bool
    constraint_results: List[ConstraintResult] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    sanity_check_results: List[Tuple] = field(default_factory=list)
    severity_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    figures: Dict[str, plt.Figure] = field(default_factory=dict)
    comparison: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        """Return a human-readable text summary."""
        lines = [
            "=" * 60,
            "SynthDrive v0.1 — Validation Report",
            "=" * 60,
            f"Policies:               {self.n_policies:,}",
        ]

        # Portfolio statistics
        stats = self.statistics
        if stats:
            lines.append(f"Total exposure:         {stats.get('total_exposure', 0):,.1f} policy-years")
            lines.append(f"Mean exposure:          {stats.get('mean_exposure', 0):.3f}")
            lines.append(f"Zero-claim pct:         {stats.get('zero_claim_pct', 0):.1%}")
            freq = stats.get("claim_count_per_policy_year")
            if freq is not None:
                lines.append(f"Frequency (ann.):       {freq:.4f}")
            sev = stats.get("mean_severity")
            if sev is not None:
                lines.append(f"Mean severity:          {sev:,.0f}")
                lines.append(f"Median severity:        {stats.get('median_severity', 0):,.0f}")
                lines.append(f"95th pct severity:      {stats.get('p95_severity', 0):,.0f}")
            pp = stats.get("mean_pure_premium")
            if pp is not None:
                lines.append(f"Mean pure premium:      {pp:,.2f}")

        lines.append("")
        lines.append(f"Constraints:            {'ALL PASSED' if self.passed_constraints else 'FAILURES DETECTED'}")

        n_fail = sum(1 for _, p, _ in self.constraint_results if not p)
        if n_fail > 0:
            lines.append(f"  {n_fail} constraint(s) failed.")
            for name, passed, msg in self.constraint_results:
                if not passed:
                    lines.append(f"  FAIL  {name}: {msg}")

        lines.append("")
        lines.append("Sanity checks:")
        for name, passed, msg in self.sanity_check_results:
            flag = "  OK  " if passed else "  WARN"
            lines.append(f"  {flag}  {name}: {msg}")

        if self.comparison is not None and len(self.comparison) > 0:
            lines.append("")
            lines.append("Distribution comparison (synthetic vs reference):")
            lines.append(
                self.comparison[
                    ["variable", "synth_mean", "ref_mean", "mean_diff_pct"]
                ].to_string(index=False)
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, output_dir: str) -> None:
        """
        Save the text report and all figures to output_dir.

        Files created:
            validation_report.txt
            constraint_report.txt
            figures/frequency_by_age.png
            figures/frequency_by_territory.png
            figures/severity_histogram.png
            figures/feature_histograms.png
            figures/correlation_heatmap.png
            figures/<extra comparison plots>.png
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_dir = out / "figures"
        fig_dir.mkdir(exist_ok=True)

        # Text report
        (out / "validation_report.txt").write_text(self.summary())
        (out / "constraint_report.txt").write_text(
            format_constraint_report(self.constraint_results)
        )

        # Figures
        for name, fig in self.figures.items():
            safe_name = name.replace(" ", "_").replace("/", "-")
            fig.savefig(fig_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")

        print(f"Report saved to {out}")

    def __repr__(self) -> str:
        return (
            f"ValidationReport("
            f"n={self.n_policies:,}, "
            f"constraints={'OK' if self.passed_constraints else 'FAIL'}, "
            f"figures={list(self.figures.keys())})"
        )


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def validate(
    df: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    make_plots: bool = True,
) -> ValidationReport:
    """
    Validate a synthetic telematics portfolio.

    Parameters
    ----------
    df : pd.DataFrame
        Output of synthdrive.generate().
    reference_data : pd.DataFrame, optional
        Reference dataset for distribution comparison.
        Typically the loaded seed CSV (seed_data.df).
    make_plots : bool
        If True, generate matplotlib figures and store in report.figures.

    Returns
    -------
    ValidationReport
    """
    # --- Constraint check
    constraint_results = check_constraints(df)
    passed = all_passed(constraint_results)

    # --- Portfolio statistics
    stats = portfolio_summary(df)
    sanity = sanity_checks(df)
    sev_summary = severity_distribution(df)

    # --- Distribution comparison
    comparison = None
    if reference_data is not None:
        comparison = compare_distributions(df, reference_data)

    # --- Figures
    figures: Dict[str, plt.Figure] = {}
    if make_plots:
        try:
            figures["frequency_by_age"] = plot_frequency_by_age(df)
        except Exception:
            pass
        try:
            figures["frequency_by_territory"] = plot_frequency_by_territory(df)
        except Exception:
            pass
        try:
            figures["severity_histogram"] = plot_severity_histogram(df)
        except Exception:
            pass
        try:
            figures["feature_histograms"] = plot_feature_histograms(df)
        except Exception:
            pass
        try:
            figures["correlation_heatmap"] = plot_correlation_heatmap(df)
        except Exception:
            pass
        if reference_data is not None:
            for var in ["insured_age", "annual_miles_drive", "claim_amount"]:
                try:
                    fig = plot_distribution_comparison(df, reference_data, var)
                    figures[f"comparison_{var}"] = fig
                except Exception:
                    pass

    return ValidationReport(
        n_policies=len(df),
        passed_constraints=passed,
        constraint_results=constraint_results,
        statistics=stats,
        sanity_check_results=sanity,
        severity_summary=sev_summary,
        figures=figures,
        comparison=comparison,
    )
