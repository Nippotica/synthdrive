"""
Schema definitions for SynthDrive v0.1.

All percentage and compositional variables are stored as proportions in [0.0, 1.0].
For example, pct_drive_mon = 0.15 means 15 percent of driving occurred on Monday.

The canonical schema covers the parameter-only (offline) operating mode.
When the So-Boucher-Valdez seed CSV is loaded, load_seed.py preserves whatever
additional time-band columns it contains (e.g. pct_drive_5hrs, pct_drive_6hrs, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class VariableSpec:
    """Specification for a single dataset column."""

    name: str
    dtype: str          # "float64", "int64", "category", "object"
    group: str          # "policy", "driver", "vehicle", "telematics", "claim"
    var_type: str       # "identifier", "continuous", "integer", "binary",
                        # "categorical", "proportion", "compositional",
                        # "rate", "response"
    description: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    categories: Optional[Tuple[str, ...]] = None
    units: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Full variable specification list
# ---------------------------------------------------------------------------

VARIABLE_SPECS: Tuple[VariableSpec, ...] = (

    # -----------------------------------------------------------------------
    # Policy / exposure
    # -----------------------------------------------------------------------
    VariableSpec(
        name="policy_id",
        dtype="object",
        group="policy",
        var_type="identifier",
        description="Unique policy identifier.",
    ),
    VariableSpec(
        name="duration",
        dtype="int64",
        group="policy",
        var_type="integer",
        description="Policy duration in days (integer). Typical values: 91, 182, 365, 366.",
        min_val=1.0,
        max_val=366.0,
        units="days",
        notes="Strictly positive integer. Matches the SBV CSV Duration column.",
    ),
    VariableSpec(
        name="exposure",
        dtype="float64",
        group="policy",
        var_type="continuous",
        description="Risk exposure in policy-years. Derived as duration / 365.",
        min_val=0.0,
        max_val=1.0,
        units="policy-years",
        notes="Strictly positive. Used as offset in frequency GLM: log(exposure).",
    ),

    # -----------------------------------------------------------------------
    # Driver variables
    # -----------------------------------------------------------------------
    VariableSpec(
        name="insured_age",
        dtype="int64",
        group="driver",
        var_type="integer",
        description="Age of the primary insured driver in years.",
        min_val=16.0,
        max_val=90.0,
        units="years",
    ),
    VariableSpec(
        name="insured_sex",
        dtype="category",
        group="driver",
        var_type="binary",
        description="Sex of the primary insured driver.",
        categories=("Male", "Female"),
    ),
    VariableSpec(
        name="marital_status",
        dtype="category",
        group="driver",
        var_type="binary",
        description="Marital status of the primary insured driver.",
        categories=("Married", "Single"),
    ),
    VariableSpec(
        name="credit_score",
        dtype="int64",
        group="driver",
        var_type="integer",
        description="Credit-based insurance score.",
        min_val=400.0,
        max_val=900.0,
        notes="Higher values indicate better creditworthiness.",
    ),
    VariableSpec(
        name="years_no_claims",
        dtype="int64",
        group="driver",
        var_type="integer",
        description="Number of consecutive claim-free years prior to this policy.",
        min_val=0.0,
        max_val=50.0,
        units="years",
    ),

    # -----------------------------------------------------------------------
    # Vehicle variables
    # -----------------------------------------------------------------------
    VariableSpec(
        name="car_age",
        dtype="int64",
        group="vehicle",
        var_type="integer",
        description="Age of the insured vehicle in years. Can be negative (model-year convention).",
        min_val=-2.0,
        max_val=20.0,
        units="years",
        notes="Observed range [-2, 20]. Negative values reflect model-year convention.",
    ),
    VariableSpec(
        name="car_use",
        dtype="category",
        group="vehicle",
        var_type="categorical",
        description="Primary use of the insured vehicle.",
        categories=("Commute", "Private", "Commercial", "Farmer"),
    ),
    VariableSpec(
        name="territory",
        dtype="int64",
        group="vehicle",
        var_type="integer",
        description="Rating territory zone code. Continuous integer in [11, 91].",
        min_val=11.0,
        max_val=91.0,
        notes="Not a categorical variable. Mean ≈ 56.5, std ≈ 24.0.",
    ),
    VariableSpec(
        name="region",
        dtype="category",
        group="vehicle",
        var_type="categorical",
        description="Broader geographic region.",
        categories=("Urban", "Rural"),
    ),
    VariableSpec(
        name="annual_miles_drive",
        dtype="float64",
        group="vehicle",
        var_type="continuous",
        description="Estimated annual miles the vehicle is driven.",
        min_val=500.0,
        max_val=60_000.0,
        units="miles/year",
    ),

    # -----------------------------------------------------------------------
    # Telematics — mileage
    # -----------------------------------------------------------------------
    VariableSpec(
        name="annual_pct_driven",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Fraction of the year during which the vehicle was driven.",
        min_val=0.0,
        max_val=1.0,
        notes="Proportion [0, 1]. Does not represent fraction of annual_miles_drive.",
    ),
    VariableSpec(
        name="total_miles_driven",
        dtype="float64",
        group="telematics",
        var_type="continuous",
        description="Total miles driven during the policy period.",
        min_val=0.0,
        units="miles",
    ),

    # -----------------------------------------------------------------------
    # Telematics — driving pattern (day of week)
    # -----------------------------------------------------------------------
    VariableSpec(
        name="avg_days_week",
        dtype="float64",
        group="telematics",
        var_type="continuous",
        description="Average number of days per week the vehicle is driven.",
        min_val=0.0,
        max_val=7.0,
        units="days/week",
    ),
    VariableSpec(
        name="pct_drive_mon",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Mondays.",
        min_val=0.0,
        max_val=1.0,
        notes="Compositional: sum(pct_drive_mon..sun) = 1.0.",
    ),
    VariableSpec(
        name="pct_drive_tue",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Tuesdays.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_wed",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Wednesdays.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_thu",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Thursdays.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_fri",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Fridays.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_sat",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Saturdays.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_sun",
        dtype="float64",
        group="telematics",
        var_type="compositional",
        description="Proportion of driving occurring on Sundays.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_wkday",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving on weekdays (Mon–Fri). Derived: sum(mon..fri).",
        min_val=0.0,
        max_val=1.0,
        notes="Derived column. pct_drive_wkday = pct_drive_mon + ... + pct_drive_fri.",
    ),
    VariableSpec(
        name="pct_drive_wkend",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving on weekends (Sat–Sun). Derived: sum(sat, sun).",
        min_val=0.0,
        max_val=1.0,
        notes="Derived column. pct_drive_wkend = pct_drive_sat + pct_drive_sun.",
    ),

    # -----------------------------------------------------------------------
    # Telematics — time of day
    # -----------------------------------------------------------------------
    VariableSpec(
        name="pct_drive_rush_am",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving during morning rush hour (approximately 7–9 am).",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_rush_pm",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving during evening rush hour (approximately 4–7 pm).",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_2hrs",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving between 2 am and 3 am.",
        min_val=0.0,
        max_val=1.0,
        notes="Late-night driving band; associated with elevated risk.",
    ),
    VariableSpec(
        name="pct_drive_3hrs",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving between 3 am and 4 am.",
        min_val=0.0,
        max_val=1.0,
    ),
    VariableSpec(
        name="pct_drive_4hrs",
        dtype="float64",
        group="telematics",
        var_type="proportion",
        description="Proportion of driving between 4 am and 5 am.",
        min_val=0.0,
        max_val=1.0,
    ),

    # -----------------------------------------------------------------------
    # Telematics — hard acceleration (raw event counts)
    # Threshold set: {6, 8, 9, 11, 12, 14} mph/s (no 10 mph/s column)
    # -----------------------------------------------------------------------
    VariableSpec(
        name="accel_6_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard acceleration raw event count at 6 mph/s threshold.",
        min_val=0.0,
        notes="Monotone: accel_6 >= accel_8 >= accel_9 >= accel_11 >= accel_12 >= accel_14.",
    ),
    VariableSpec(
        name="accel_8_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard acceleration raw event count at 8 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="accel_9_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard acceleration raw event count at 9 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="accel_11_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard acceleration raw event count at 11 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="accel_12_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard acceleration raw event count at 12 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="accel_14_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard acceleration raw event count at 14 mph/s threshold.",
        min_val=0.0,
    ),

    # -----------------------------------------------------------------------
    # Telematics — hard braking (raw event counts)
    # Threshold set: {6, 8, 9, 11, 12, 14} mph/s (no 10 mph/s column)
    # -----------------------------------------------------------------------
    VariableSpec(
        name="brake_6_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard braking raw event count at 6 mph/s threshold.",
        min_val=0.0,
        notes="Monotone: brake_6 >= brake_8 >= brake_9 >= brake_11 >= brake_12 >= brake_14.",
    ),
    VariableSpec(
        name="brake_8_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard braking raw event count at 8 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="brake_9_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard braking raw event count at 9 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="brake_11_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard braking raw event count at 11 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="brake_12_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard braking raw event count at 12 mph/s threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="brake_14_miles",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Hard braking raw event count at 14 mph/s threshold.",
        min_val=0.0,
    ),

    # -----------------------------------------------------------------------
    # Telematics — left turn intensity (raw event counts, not proportions)
    # -----------------------------------------------------------------------
    VariableSpec(
        name="left_turn_intensity_08",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Left turn raw event count at 0.8 g lateral acceleration threshold.",
        min_val=0.0,
        notes="Monotone: left_08 >= left_09 >= left_10 >= left_11 >= left_12.",
    ),
    VariableSpec(
        name="left_turn_intensity_09",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Left turn raw event count at 0.9 g lateral acceleration threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="left_turn_intensity_10",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Left turn raw event count at 1.0 g lateral acceleration threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="left_turn_intensity_11",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Left turn raw event count at 1.1 g lateral acceleration threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="left_turn_intensity_12",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Left turn raw event count at 1.2 g lateral acceleration threshold.",
        min_val=0.0,
    ),

    # -----------------------------------------------------------------------
    # Telematics — right turn intensity (raw event counts, not proportions)
    # -----------------------------------------------------------------------
    VariableSpec(
        name="right_turn_intensity_08",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Right turn raw event count at 0.8 g lateral acceleration threshold.",
        min_val=0.0,
        notes="Monotone: right_08 >= right_09 >= right_10 >= right_11 >= right_12.",
    ),
    VariableSpec(
        name="right_turn_intensity_09",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Right turn raw event count at 0.9 g lateral acceleration threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="right_turn_intensity_10",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Right turn raw event count at 1.0 g lateral acceleration threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="right_turn_intensity_11",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Right turn raw event count at 1.1 g lateral acceleration threshold.",
        min_val=0.0,
    ),
    VariableSpec(
        name="right_turn_intensity_12",
        dtype="int64",
        group="telematics",
        var_type="integer",
        description="Right turn raw event count at 1.2 g lateral acceleration threshold.",
        min_val=0.0,
    ),

    # -----------------------------------------------------------------------
    # Claim response variables
    # -----------------------------------------------------------------------
    VariableSpec(
        name="claim_count",
        dtype="int64",
        group="claim",
        var_type="response",
        description="Number of at-fault claims during the policy period.",
        min_val=0.0,
        notes="Non-negative integer. Many policies have claim_count = 0.",
    ),
    VariableSpec(
        name="claim_amount",
        dtype="float64",
        group="claim",
        var_type="response",
        description="Aggregate claim amount for the policy period.",
        min_val=0.0,
        units="currency",
        notes=(
            "Zero when claim_count = 0. "
            "Positive when claim_count > 0. "
            "Currency units are not denominated in a specific currency."
        ),
    ),
    VariableSpec(
        name="pure_premium",
        dtype="float64",
        group="claim",
        var_type="response",
        description="Pure premium: claim_amount / exposure.",
        min_val=0.0,
        units="currency/policy-year",
        notes=(
            "Zero when claim_count = 0. "
            "Defined as claim_amount / exposure. "
            "Represents expected annual loss cost."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

# Map from variable name to its spec
SPEC_BY_NAME: dict[str, VariableSpec] = {s.name: s for s in VARIABLE_SPECS}

# Ordered column name list for the canonical parameter-mode schema
CANONICAL_COLUMNS: Tuple[str, ...] = tuple(s.name for s in VARIABLE_SPECS)

# Grouped column lists
POLICY_COLUMNS = tuple(s.name for s in VARIABLE_SPECS if s.group == "policy")
DRIVER_COLUMNS = tuple(s.name for s in VARIABLE_SPECS if s.group == "driver")
VEHICLE_COLUMNS = tuple(s.name for s in VARIABLE_SPECS if s.group == "vehicle")
TELEMATICS_COLUMNS = tuple(s.name for s in VARIABLE_SPECS if s.group == "telematics")
CLAIM_COLUMNS = tuple(s.name for s in VARIABLE_SPECS if s.group == "claim")

# Day-of-week compositional variables (must sum to 1.0)
DOW_COLUMNS = (
    "pct_drive_mon",
    "pct_drive_tue",
    "pct_drive_wed",
    "pct_drive_thu",
    "pct_drive_fri",
    "pct_drive_sat",
    "pct_drive_sun",
)

# Accel/brake threshold sequences (must be non-increasing)
# Threshold set: {6, 8, 9, 11, 12, 14} mph/s — no 10 mph/s column
ACCEL_COLUMNS = (
    "accel_6_miles",
    "accel_8_miles",
    "accel_9_miles",
    "accel_11_miles",
    "accel_12_miles",
    "accel_14_miles",
)
BRAKE_COLUMNS = (
    "brake_6_miles",
    "brake_8_miles",
    "brake_9_miles",
    "brake_11_miles",
    "brake_12_miles",
    "brake_14_miles",
)
LEFT_TURN_COLUMNS = (
    "left_turn_intensity_08",
    "left_turn_intensity_09",
    "left_turn_intensity_10",
    "left_turn_intensity_11",
    "left_turn_intensity_12",
)
RIGHT_TURN_COLUMNS = (
    "right_turn_intensity_08",
    "right_turn_intensity_09",
    "right_turn_intensity_10",
    "right_turn_intensity_11",
    "right_turn_intensity_12",
)
