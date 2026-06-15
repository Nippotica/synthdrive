"""
Seed CSV loader for SynthDrive v0.1.

Design principle (Q1)
----------------------
SynthDrive v0.1 can operate in two modes:

    Local seed mode:  a CSV path is supplied → load_sbv_csv() reads it and
                      returns a SeedData object.

    Parameter mode:   no CSV path is supplied (or the path is None) →
                      all sampling uses calibrated defaults from CorePreset.

There is no automatic network download and no hard failure when the CSV is
absent.  A fetch helper may be added in a future release.

SBV column mapping (Q3)
------------------------
The So–Boucher–Valdez dataset uses R-style column names with dots and mixed
capitalisation.  load_sbv_csv() remaps these to SynthDrive's snake_case
conventions.  Any additional time-band columns present in the CSV (e.g.
pct_drive_5hrs, pct_drive_6hrs, ...) are preserved rather than discarded.

Proportion convention (Q2)
---------------------------
All percentage/proportion columns are checked and normalised to [0.0, 1.0].
If the loaded CSV contains values in [0, 100] range, they are scaled down
automatically with a warning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SBV → SynthDrive column name mapping
# ---------------------------------------------------------------------------

# Canonical SBV R column names → SynthDrive snake_case names.
# Keys that are absent from the loaded CSV are ignored silently.
SBV_RENAME_MAP: Dict[str, str] = {
    # Policy / exposure
    "Duration":                "duration",

    # Driver
    "Insured.age":             "insured_age",
    "Insured.sex":             "insured_sex",
    "Marital":                 "marital_status",
    "Credit.score":            "credit_score",
    "Years.noclaims":          "years_no_claims",

    # Vehicle
    "Car.age":                 "car_age",
    "Car.use":                 "car_use",
    "Territory":               "territory",
    "Region":                  "region",
    "Annual.miles.drive":      "annual_miles_drive",

    # Telematics — mileage
    "Annual.pct.driven":       "annual_pct_driven",
    "Total.miles.driven":      "total_miles_driven",

    # Telematics — driving pattern
    "Avgdays.week":            "avg_days_week",
    "Pct.drive.mon":           "pct_drive_mon",
    "Pct.drive.tue":           "pct_drive_tue",
    "Pct.drive.wed":           "pct_drive_wed",
    "Pct.drive.thr":           "pct_drive_thu",
    "Pct.drive.fri":           "pct_drive_fri",
    "Pct.drive.sat":           "pct_drive_sat",
    "Pct.drive.sun":           "pct_drive_sun",
    "Pct.drive.wkday":         "pct_drive_wkday",
    "Pct.drive.wkend":         "pct_drive_wkend",

    # Telematics — time of day (canonical parameter-mode set)
    "Pct.drive.rush am":       "pct_drive_rush_am",
    "Pct.drive.rush pm":       "pct_drive_rush_pm",
    "Pct.drive.2hrs":          "pct_drive_2hrs",
    "Pct.drive.3hrs":          "pct_drive_3hrs",
    "Pct.drive.4hrs":          "pct_drive_4hrs",

    # Telematics — hard acceleration (raw event counts)
    # Threshold set: {6, 8, 9, 11, 12, 14} mph/s (zero-padded column names)
    "Accel.06miles":           "accel_6_miles",
    "Accel.08miles":           "accel_8_miles",
    "Accel.09miles":           "accel_9_miles",
    "Accel.11miles":           "accel_11_miles",
    "Accel.12miles":           "accel_12_miles",
    "Accel.14miles":           "accel_14_miles",

    # Telematics — hard braking (raw event counts)
    "Brake.06miles":           "brake_6_miles",
    "Brake.08miles":           "brake_8_miles",
    "Brake.09miles":           "brake_9_miles",
    "Brake.11miles":           "brake_11_miles",
    "Brake.12miles":           "brake_12_miles",
    "Brake.14miles":           "brake_14_miles",

    # Telematics — turn intensity (raw event counts)
    "Left.turn.intensity08":   "left_turn_intensity_08",
    "Left.turn.intensity09":   "left_turn_intensity_09",
    "Left.turn.intensity10":   "left_turn_intensity_10",
    "Left.turn.intensity11":   "left_turn_intensity_11",
    "Left.turn.intensity12":   "left_turn_intensity_12",
    "Right.turn.intensity08":  "right_turn_intensity_08",
    "Right.turn.intensity09":  "right_turn_intensity_09",
    "Right.turn.intensity10":  "right_turn_intensity_10",
    "Right.turn.intensity11":  "right_turn_intensity_11",
    "Right.turn.intensity12":  "right_turn_intensity_12",

    # Claims (response variables)
    "NB_Claim":                "claim_count",
    "AMT_Claim":               "claim_amount",
}

# Proportion columns that must be on [0.0, 1.0] after loading.
# If values exceed 1.0 (i.e. stored as 0–100 in the CSV), they are divided by 100.
# Turn intensity columns are excluded: they are raw event counts, not proportions.
PROPORTION_COLUMNS: Tuple[str, ...] = (
    "annual_pct_driven",
    "pct_drive_mon", "pct_drive_tue", "pct_drive_wed",
    "pct_drive_thu", "pct_drive_fri", "pct_drive_sat", "pct_drive_sun",
    "pct_drive_wkday", "pct_drive_wkend",
    "pct_drive_rush_am", "pct_drive_rush_pm",
    "pct_drive_2hrs", "pct_drive_3hrs", "pct_drive_4hrs",
)

# Pattern for additional time-of-day band columns to preserve from the seed CSV.
# Any column whose name matches Pct.drive.Nhrs (N = integer) is kept.
import re
_EXTRA_TIME_BAND_PATTERN = re.compile(r"^Pct\.drive\.(\d+)hrs$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# SeedData container
# ---------------------------------------------------------------------------


@dataclass
class SeedData:
    """
    Container for a loaded seed dataset.

    Attributes
    ----------
    df : pd.DataFrame
        Cleaned seed DataFrame with SynthDrive column names.
        All proportion columns are on [0.0, 1.0].
    source_path : str
        Path to the original CSV file.
    n_rows : int
        Number of rows in the loaded dataset.
    extra_time_bands : Tuple[str, ...]
        Any additional time-of-day band columns (e.g. pct_drive_5hrs) that
        were found in the seed CSV beyond the canonical parameter-mode set.
    column_map : Dict[str, str]
        Record of the original → renamed column mappings applied.
    """

    df: pd.DataFrame
    source_path: str
    n_rows: int
    extra_time_bands: Tuple[str, ...] = field(default_factory=tuple)
    column_map: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SeedData(n_rows={self.n_rows}, "
            f"n_cols={len(self.df.columns)}, "
            f"source='{self.source_path}')"
        )


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------


def _detect_extra_time_bands(df_raw: pd.DataFrame) -> Dict[str, str]:
    """
    Detect and return a rename mapping for any additional time-band columns
    (beyond the canonical set) present in the raw CSV.

    Returns {original_name: snake_case_name}.
    """
    canonical_time_bands = {
        "Pct.drive.2hrs", "Pct.drive.3hrs", "Pct.drive.4hrs",
        "Pct.drive.rush am", "Pct.drive.rush pm",
    }
    extra: Dict[str, str] = {}
    for col in df_raw.columns:
        if col in canonical_time_bands:
            continue
        m = _EXTRA_TIME_BAND_PATTERN.match(col)
        if m:
            n_hour = m.group(1)
            snake = f"pct_drive_{n_hour}hrs"
            extra[col] = snake
    return extra


def _normalise_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect proportion columns stored as 0–100 and divide by 100.

    A column is considered 0–100 if its 95th percentile exceeds 1.5.
    A warning is issued for each such column.
    """
    df = df.copy()
    all_prop_cols = list(PROPORTION_COLUMNS)
    # Also check any extra pct_drive_Nhrs columns
    extra = [c for c in df.columns if re.match(r"^pct_drive_\d+hrs$", c)]
    all_prop_cols += extra

    for col in all_prop_cols:
        if col not in df.columns:
            continue
        col_data = pd.to_numeric(df[col], errors="coerce")
        p95 = col_data.quantile(0.95)
        if p95 > 1.5:
            warnings.warn(
                f"Column '{col}' appears to be stored as 0–100 (p95 = {p95:.1f}). "
                "Dividing by 100 to convert to [0, 1] proportion convention.",
                UserWarning,
                stacklevel=3,
            )
            df[col] = col_data / 100.0
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0, upper=1.0)

    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_sbv_csv(
    path: str | Path,
    *,
    nrows: Optional[int] = None,
    low_memory: bool = False,
) -> SeedData:
    """
    Load a So–Boucher–Valdez style telematics CSV file.

    The function:
    1. Reads the CSV from the supplied local path (no network access).
    2. Renames columns from SBV R-style to SynthDrive snake_case.
    3. Preserves any extra time-band columns (e.g. pct_drive_5hrs).
    4. Normalises proportion columns to [0.0, 1.0] if needed.
    5. Returns a SeedData container.

    Parameters
    ----------
    path : str or Path
        Local filesystem path to the CSV file.
    nrows : int, optional
        If provided, read only the first nrows rows.  Useful for testing.
    low_memory : bool
        Passed to pandas.read_csv.

    Returns
    -------
    SeedData

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as a CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Seed CSV not found: {path}\n"
            "SynthDrive can still run in parameter mode without a seed file. "
            "Pass seed_path=None to generate() to use built-in default parameters."
        )

    try:
        df_raw = pd.read_csv(path, nrows=nrows, low_memory=low_memory)
    except Exception as exc:
        raise ValueError(f"Could not parse seed CSV at {path}: {exc}") from exc

    # Detect extra time-band columns before rename
    extra_band_map = _detect_extra_time_bands(df_raw)

    # Build full rename map (canonical + extra time bands)
    full_rename = {**SBV_RENAME_MAP, **extra_band_map}
    # Only apply mappings for columns that exist
    applicable_rename = {k: v for k, v in full_rename.items() if k in df_raw.columns}
    df = df_raw.rename(columns=applicable_rename)

    # Normalise proportion columns to [0, 1]
    df = _normalise_proportions(df)

    # Compute exposure as duration (days) / 365
    if "exposure" not in df.columns and "duration" in df.columns:
        duration_days = pd.to_numeric(df["duration"], errors="coerce")
        df["exposure"] = (duration_days / 365.0).clip(lower=1e-6, upper=1.0)

    extra_time_bands = tuple(
        applicable_rename[orig] for orig in extra_band_map
        if orig in applicable_rename
    )

    return SeedData(
        df=df,
        source_path=str(path),
        n_rows=len(df),
        extra_time_bands=extra_time_bands,
        column_map=applicable_rename,
    )


# ---------------------------------------------------------------------------
# Safe load helper (returns None on failure)
# ---------------------------------------------------------------------------


def try_load_seed(path: Optional[str | Path]) -> Optional[SeedData]:
    """
    Attempt to load a seed CSV.  Return None if path is None or loading fails.

    This is the function called by generate() so that a missing or invalid
    seed file never raises a hard error — the caller falls through to
    parameter mode.

    Parameters
    ----------
    path : str, Path, or None

    Returns
    -------
    SeedData or None
    """
    if path is None:
        return None
    try:
        return load_sbv_csv(path)
    except FileNotFoundError as exc:
        warnings.warn(str(exc), UserWarning, stacklevel=2)
        return None
    except Exception as exc:
        warnings.warn(
            f"Failed to load seed CSV from '{path}': {exc}. "
            "Falling back to parameter mode.",
            UserWarning,
            stacklevel=2,
        )
        return None
