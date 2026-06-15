"""Tests for synthdrive.validate."""

import numpy as np
import pandas as pd
import pytest

import synthdrive
from synthdrive.validate.diagnostics import (
    frequency_by_band,
    portfolio_summary,
    sanity_checks,
    severity_distribution,
)
from synthdrive.validate.report import validate


# ---------------------------------------------------------------------------
# portfolio_summary
# ---------------------------------------------------------------------------


def test_portfolio_summary_returns_dict(small_df):
    stats = portfolio_summary(small_df)
    assert isinstance(stats, dict)
    assert "n_policies" in stats
    assert stats["n_policies"] == len(small_df)


def test_portfolio_summary_frequency_key(small_df):
    stats = portfolio_summary(small_df)
    assert "claim_count_per_policy_year" in stats
    assert stats["claim_count_per_policy_year"] >= 0


def test_portfolio_summary_severity_key(small_df):
    stats = portfolio_summary(small_df)
    if stats.get("mean_severity") is not None:
        assert stats["mean_severity"] > 0


# ---------------------------------------------------------------------------
# frequency_by_band
# ---------------------------------------------------------------------------


def test_frequency_by_band_returns_dataframe(small_df):
    result = frequency_by_band(small_df, "insured_age")
    assert isinstance(result, pd.DataFrame)
    assert "frequency" in result.columns


def test_frequency_by_band_raises_on_missing_column(small_df):
    with pytest.raises(ValueError):
        frequency_by_band(small_df, "nonexistent_column_xyz")


def test_frequency_by_band_n_bands(small_df):
    result = frequency_by_band(small_df, "insured_age", n_bands=4)
    assert len(result) <= 4  # may be fewer if quantiles coincide


# ---------------------------------------------------------------------------
# severity_distribution
# ---------------------------------------------------------------------------


def test_severity_distribution_returns_dataframe(small_df):
    result = severity_distribution(small_df)
    assert isinstance(result, pd.DataFrame)
    assert "statistic" in result.columns
    assert "value" in result.columns


def test_severity_distribution_stats_positive(small_df):
    result = severity_distribution(small_df)
    mean_row = result[result["statistic"] == "mean"]
    if len(mean_row) > 0 and not np.isnan(mean_row["value"].iloc[0]):
        assert mean_row["value"].iloc[0] > 0


# ---------------------------------------------------------------------------
# sanity_checks
# ---------------------------------------------------------------------------


def test_sanity_checks_returns_list(small_df):
    results = sanity_checks(small_df)
    assert isinstance(results, list)
    assert all(len(r) == 3 for r in results)


def test_sanity_checks_frequency_in_range(medium_df):
    results = sanity_checks(medium_df)
    freq_check = next(
        (r for r in results if "frequency" in r[0]), None
    )
    if freq_check:
        assert freq_check[1], f"Frequency sanity check failed: {freq_check[2]}"


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_returns_report(small_df):
    report = validate(small_df, make_plots=False)
    assert report is not None
    assert report.n_policies == len(small_df)


def test_validate_passed_constraints(small_df):
    report = validate(small_df, make_plots=False)
    assert report.passed_constraints, (
        "Constraint failures:\n"
        + "\n".join(f"  FAIL {n}: {m}" for n, p, m in report.constraint_results if not p)
    )


def test_validate_summary_string(small_df):
    report = validate(small_df, make_plots=False)
    summary = report.summary()
    assert isinstance(summary, str)
    assert "SynthDrive" in summary
    assert "Policies" in summary


def test_validate_with_plots(small_df):
    report = validate(small_df, make_plots=True)
    assert len(report.figures) > 0
    # All returned figures should be matplotlib Figure objects
    import matplotlib.pyplot as plt
    for name, fig in report.figures.items():
        assert isinstance(fig, plt.Figure), f"'{name}' is not a Figure"


def test_validate_save(small_df, tmp_path):
    report = validate(small_df, make_plots=True)
    report.save(str(tmp_path))
    assert (tmp_path / "validation_report.txt").exists()
    assert (tmp_path / "constraint_report.txt").exists()
    fig_dir = tmp_path / "figures"
    assert fig_dir.is_dir()
    assert len(list(fig_dir.glob("*.png"))) > 0


def test_validate_with_reference_data(small_df):
    """Using the generated data as its own reference should give ~0 differences."""
    report = validate(small_df, reference_data=small_df, make_plots=False)
    if report.comparison is not None and len(report.comparison) > 0:
        max_diff = report.comparison["mean_diff_pct"].abs().max()
        assert max_diff < 1e-6, f"Self-comparison shows non-zero diff: {max_diff}"


def test_validate_repr(small_df):
    report = validate(small_df, make_plots=False)
    repr_str = repr(report)
    assert "ValidationReport" in repr_str
