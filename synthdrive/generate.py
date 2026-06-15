"""
Main generation entry point for SynthDrive v0.1.

Public API
----------
    df = synthdrive.generate(n=100_000, random_state=42)

    # With a local seed CSV
    df = synthdrive.generate(n=100_000, seed_path="/data/sbv.csv", random_state=42)

    # With a custom preset (parameter-mode only in v0.1)
    df = synthdrive.generate(n=50_000, preset="core", random_state=0)

Dual-mode design
----------------
If seed_path is provided and the file exists, the copula and marginal
distributions are fit from the seed data (seed mode).

If seed_path is None or the file cannot be loaded, all sampling uses the
calibrated default parameters from the named preset (parameter mode).

No network access is performed in either mode.

Column order
------------
The returned DataFrame follows the canonical schema order defined in
synthdrive.data.schema.CANONICAL_COLUMNS.  When running in seed mode, any
extra time-band columns from the seed CSV are appended after the canonical
columns.

policy_id format
----------------
"P{zero-padded 8-digit integer}", e.g. "P00000001".
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from synthdrive.claims.frequency import FrequencyModel
from synthdrive.claims.premium import compute_pure_premium
from synthdrive.claims.severity import SeverityModel
from synthdrive.data.load_seed import try_load_seed
from synthdrive.data.schema import CANONICAL_COLUMNS
from synthdrive.features.constraints import enforce_all_constraints
from synthdrive.features.sampler import FeatureSampler
from synthdrive.presets.core import load_core_preset


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

_PRESET_LOADERS = {
    "core": load_core_preset,
}


def _load_preset(name: str) -> object:
    if name not in _PRESET_LOADERS:
        available = list(_PRESET_LOADERS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    return _PRESET_LOADERS[name]()


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def generate(
    n: int = 100_000,
    preset: str = "core",
    seed_path: Optional[str] = None,
    random_state: Optional[int] = None,
    add_policy_id: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic telematics portfolio.

    Parameters
    ----------
    n : int
        Number of policies to generate.  Must be >= 1.
    preset : str
        Parameter preset to use.  "core" is the only supported preset in v0.1.
    seed_path : str or None
        Path to a local So–Boucher–Valdez style seed CSV.

        - If provided and the file exists, the copula and marginal distributions
          are fit from the seed data (seed mode).
        - If None or the file cannot be read, calibrated default parameters are
          used (parameter mode).  No error is raised.
    random_state : int or None
        Seed for numpy's default_rng.  Pass an integer for reproducible output.
        None gives a non-deterministic run.
    add_policy_id : bool
        If True (default), add a "policy_id" column as the first column.

    Returns
    -------
    pd.DataFrame with n rows and all canonical columns plus any extra time-band
    columns from the seed CSV when running in seed mode.

    Notes
    -----
    All proportion variables are on [0.0, 1.0].
    See synthdrive.data.schema for full column documentation.

    Examples
    --------
    >>> import synthdrive
    >>> df = synthdrive.generate(n=10_000, random_state=0)
    >>> df.shape[0]
    10000
    >>> "claim_count" in df.columns
    True
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    rng = np.random.default_rng(random_state)
    params = _load_preset(preset)

    # --- Attempt to load seed CSV (never raises; returns None on failure)
    seed_data = try_load_seed(seed_path)
    if seed_path is not None and seed_data is None:
        warnings.warn(
            f"Could not load seed CSV from '{seed_path}'. "
            "Running in parameter mode.",
            UserWarning,
            stacklevel=2,
        )

    mode = "seed" if seed_data is not None else "parameter"

    # --- Build sampler
    sampler = FeatureSampler(params=params, seed_data=seed_data)

    # --- Sample features (X)
    df = sampler.sample(n=n, rng=rng)

    # --- Sample claim counts
    freq_model = FrequencyModel(params=params.frequency)
    claim_counts = freq_model.sample(df=df, rng=rng)
    df["claim_count"] = claim_counts

    # --- Sample claim amounts (A)
    sev_model = SeverityModel(params=params.severity)
    df["claim_amount"] = sev_model.sample(df=df, rng=rng, claim_counts=claim_counts)

    # --- Compute pure premium
    df["pure_premium"] = compute_pure_premium(df)

    # --- Final constraint pass (claim consistency + pure premium)
    df = enforce_all_constraints(df)

    # --- Add policy IDs
    if add_policy_id:
        df.insert(0, "policy_id", [f"P{i:08d}" for i in range(1, n + 1)])

    # --- Reorder to canonical schema where possible
    df = _reorder_columns(df, seed_data)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------


def _reorder_columns(
    df: pd.DataFrame,
    seed_data: Optional[object],
) -> pd.DataFrame:
    """
    Reorder columns to follow CANONICAL_COLUMNS, then append any extras.

    Extra columns are those present in df but not in the canonical schema
    (typically extra time-band columns from a seed CSV).
    """
    canonical_present = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in set(CANONICAL_COLUMNS)]

    # Extra time-band columns from seed go right after pct_drive_4hrs
    if seed_data is not None and hasattr(seed_data, "extra_time_bands"):
        extra_time = [c for c in seed_data.extra_time_bands if c in extra_cols]
        other_extra = [c for c in extra_cols if c not in extra_time]
        extra_ordered = extra_time + other_extra
    else:
        extra_ordered = extra_cols

    return df[canonical_present + extra_ordered]
