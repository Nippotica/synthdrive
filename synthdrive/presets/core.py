"""
Calibrated default parameters for the 'core' preset of SynthDrive v0.1.

These parameters are calibrated to produce a plausible personal auto portfolio
consistent with the structure and statistics described in:

    So, Boucher, and Valdez (2021). Synthetic Dataset Generation of Driver
    Telematics. Risks 9(4): 58.

The original SBV dataset was generated from a Canadian insurer's proprietary
telematics data via extended SMOTE and feedforward neural networks.  SynthDrive
does not reproduce that pipeline.  These parameters are SynthDrive's own
calibrated approximations, informed by the paper's summary statistics.

Currency units are not denominated in any specific currency.  Values are
broadly consistent with Canadian dollar magnitudes for personal auto insurance.

Claim frequency: approximately 6 percent per policy-year (base rate).
Claim severity: mean aggregate severity approximately 3 000 monetary units
    for a single-claim policy.

All proportion variables are stored on [0.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Sub-parameter classes
# ---------------------------------------------------------------------------


@dataclass
class ExposureParams:
    """Parameters governing policy duration and exposure."""

    # Duration is drawn from a discrete mixture representing common policy terms.
    # Values in days; probabilities must sum to 1. Exposure = duration / 365.
    duration_values: Tuple[float, ...] = (365, 182, 91)
    duration_probs: Tuple[float, ...] = (0.60, 0.30, 0.10)


@dataclass
class DriverParams:
    """Parameters governing driver-level risk variables."""

    # Age: truncated normal
    age_mean: float = 45.0
    age_std: float = 14.0
    age_min: float = 16.0
    age_max: float = 90.0

    # Sex: proportion Male (from SBV empirical proportions)
    p_male: float = 0.539

    # Marital status: proportion Married (from SBV empirical proportions)
    p_married: float = 0.699

    # Credit score: truncated normal, rounded to integer
    credit_mean: float = 650.0
    credit_std: float = 85.0
    credit_min: float = 400.0
    credit_max: float = 900.0

    # Years no-claims: floor of exponential, clipped to [0, 50]
    # Mean of the underlying exponential distribution
    years_no_claims_mean: float = 4.0
    years_no_claims_max: int = 50


@dataclass
class VehicleParams:
    """Parameters governing vehicle-level risk variables."""

    # Car age: shifted gamma, clipped to [-2, 20]
    # car_age = Gamma(shape, scale) - 2 (shift of -2)
    car_age_shape: float = 2.5
    car_age_scale: float = 3.2
    car_age_min: int = -2
    car_age_max: int = 20

    # Car use: string category proportions (from SBV empirical proportions)
    car_use_probs: Dict[str, float] = field(default_factory=lambda: {
        "Commute":    0.498,
        "Private":    0.461,
        "Commercial": 0.026,
        "Farmer":     0.014,
    })

    # Territory: continuous integer zone code, range 11–91.
    # In parameter mode sampled uniformly over integers 11–91.
    territory_min: int = 11
    territory_max: int = 91

    # Region: string category proportions (from SBV empirical proportions)
    region_probs: Dict[str, float] = field(default_factory=lambda: {
        "Urban": 0.781,
        "Rural": 0.219,
    })

    # Annual miles driven: log-normal
    # log_mean and log_std of the underlying normal distribution
    annual_miles_log_mean: float = 9.35   # exp(9.35) ≈ 11 500 miles/year
    annual_miles_log_std: float = 0.42


@dataclass
class TelematicsParams:
    """Parameters governing telematics-derived variables."""

    # Average days driven per week: truncated normal clipped to [0, 7]
    avg_days_mean: float = 4.6
    avg_days_std: float = 1.2

    # Annual fraction of year the vehicle is driven: Beta(alpha, beta)
    annual_pct_alpha: float = 2.2
    annual_pct_beta: float = 4.5

    # Day-of-week Dirichlet concentration parameters [mon, tue, wed, thu, fri, sat, sun]
    # Higher values → stronger pull toward equal proportions.
    # Weekdays slightly higher than weekends.
    weekday_dirichlet: Tuple[float, ...] = (
        1.45, 1.52, 1.55, 1.53, 1.60, 1.35, 1.20,
    )

    # Rush-hour proportions: Beta(alpha, beta)
    rush_am_alpha: float = 3.0
    rush_am_beta: float = 12.0
    rush_pm_alpha: float = 4.2
    rush_pm_beta: float = 10.0

    # Late-night driving time bands (2am, 3am, 4am): Beta(alpha, beta)
    # These are typically very small proportions.
    night_2hrs_alpha: float = 0.6
    night_2hrs_beta: float = 12.0
    night_3hrs_alpha: float = 0.5
    night_3hrs_beta: float = 14.0
    night_4hrs_alpha: float = 0.5
    night_4hrs_beta: float = 14.0

    # Hard acceleration: log-normal parameters for accel_9_miles (raw event counts).
    # From SBV Accel.09miles: mean ≈ 1.75. With sigma=1.20: mu = log(1.75) - 1.20^2/2 ≈ -0.16.
    accel_9_log_mean: float = -0.16
    accel_9_log_std: float = 1.20

    # Hard braking: log-normal parameters for brake_9_miles (raw event counts).
    # From SBV Brake.09miles: mean ≈ 3.10. With sigma=0.85: mu = log(3.10) - 0.85^2/2 ≈ 0.77.
    brake_9_log_mean: float = 0.77
    brake_9_log_std: float = 0.85

    # Multipliers for deriving accel thresholds from accel_9 anchor.
    # accel_k = accel_9 * ratio_k * exp(noise), rounded to nearest integer.
    # Threshold set: {6, 8, 9, 11, 12, 14} mph/s — no 10 mph/s column.
    # Ratios calibrated from SBV empirical means: mean(Accel.kXmiles) / mean(Accel.09miles).
    accel_ratios: Dict[str, float] = field(default_factory=lambda: {
        "accel_6_miles":  24.58,  # lower threshold → more events
        "accel_8_miles":   2.58,
        "accel_9_miles":   1.00,  # anchor
        "accel_11_miles":  0.53,
        "accel_12_miles":  0.30,
        "accel_14_miles":  0.20,
    })
    brake_ratios: Dict[str, float] = field(default_factory=lambda: {
        "brake_6_miles":  26.96,  # lower threshold → more events
        "brake_8_miles":   3.09,
        "brake_9_miles":   1.00,  # anchor
        "brake_11_miles":  0.43,
        "brake_12_miles":  0.19,
        "brake_14_miles":  0.11,
    })

    # Log-normal noise added to each derived accel/brake threshold
    threshold_noise_std: float = 0.15

    # Left turn intensity base: log-normal for left_turn_intensity_08 (raw counts).
    # From SBV: mean ≈ 916. With sigma=1.50: mu = log(916) - 1.5^2/2 ≈ 5.70.
    left_08_log_mean: float = 5.70
    left_08_log_std: float = 1.50

    # Right turn intensity base: log-normal for right_turn_intensity_08 (raw counts).
    # From SBV: mean ≈ 2006. With sigma=1.50: mu = log(2006) - 1.5^2/2 ≈ 6.48.
    right_08_log_mean: float = 6.48
    right_08_log_std: float = 1.50

    # Multipliers for turn intensity sub-thresholds (applied successively)
    turn_intensity_step_ratios: Tuple[float, ...] = (1.0, 0.62, 0.42, 0.28, 0.18)
    turn_intensity_noise_std: float = 0.12

    # Correlation between left and right turn intensities (Spearman)
    left_right_turn_spearman: float = 0.72

    # Correlation between accel and brake (Spearman)
    accel_brake_spearman: float = 0.55


@dataclass
class CopulaParams:
    """
    Spearman rank correlation matrix for the joint distribution of continuous
    feature variables sampled through the Gaussian copula.

    Variables (in order):
        0: insured_age
        1: credit_score
        2: years_no_claims
        3: car_age
        4: annual_miles_drive (log scale internally)
        5: avg_days_week
        6: annual_pct_driven
        7: accel_9_miles (log scale internally)
        8: brake_9_miles (log scale internally)
    """

    # Variable names in copula order
    copula_vars: Tuple[str, ...] = (
        "insured_age",
        "credit_score",
        "years_no_claims",
        "car_age",
        "annual_miles_drive",
        "avg_days_week",
        "annual_pct_driven",
        "accel_9_miles",
        "brake_9_miles",
    )

    # Spearman rank correlation matrix (symmetric, diagonal = 1).
    # Calibrated to produce plausible joint distributions.
    # Entries are rounded to two decimal places for clarity.
    spearman_corr: Tuple[Tuple[float, ...], ...] = (
        # age  cred  yncl  cage  mile  days  pct   acc9  brk9
        ( 1.00, 0.28, 0.55, 0.08,-0.06, 0.00,-0.06,-0.08,-0.08),  # age
        ( 0.28, 1.00, 0.18,-0.05, 0.07, 0.03, 0.05,-0.10,-0.10),  # credit
        ( 0.55, 0.18, 1.00,-0.06,-0.08, 0.00,-0.05,-0.12,-0.12),  # yncl
        ( 0.08,-0.05,-0.06, 1.00,-0.12,-0.08,-0.05, 0.05, 0.05),  # cage
        (-0.06, 0.07,-0.08,-0.12, 1.00, 0.42, 0.34, 0.08, 0.10),  # miles
        ( 0.00, 0.03, 0.00,-0.08, 0.42, 1.00, 0.28, 0.05, 0.06),  # days
        (-0.06, 0.05,-0.05,-0.05, 0.34, 0.28, 1.00, 0.06, 0.07),  # pct
        (-0.08,-0.10,-0.12, 0.05, 0.08, 0.05, 0.06, 1.00, 0.55),  # accel9
        (-0.08,-0.10,-0.12, 0.05, 0.10, 0.06, 0.07, 0.55, 1.00),  # brake9
    )


@dataclass
class FrequencyParams:
    """
    Parameters for the zero-inflated negative binomial claim-count model.

    Model structure:
        N_i | not_structural_zero ~ NegBin(mu_i, alpha)
        log(mu_i) = log(exposure_i) + log(base_rate) + risk_score(X_i)
        P(structural_zero_i) = expit(gamma_0 + gamma_ynclaims * years_no_claims_i
                                     + gamma_credit * credit_norm_i)
    """

    # Base annual claim rate (per policy-year) for the reference driver.
    # Joint calibration via grid search: overall freq ≈ 0.0523, 16-25 freq ≈ 0.0849.
    base_rate: float = 0.0165

    # Negative binomial overdispersion parameter alpha
    # Var[N] = mu + alpha * mu^2
    alpha: float = 0.30

    # Zero-inflation logistic intercept
    # expit(-1.735) ≈ 0.15 → 15% structural zero probability for reference driver
    gamma_0: float = -1.735

    # Zero-inflation: effect of years_no_claims
    # More claim-free years → more likely to be a structural zero (very careful driver)
    gamma_ynclaims: float = 0.035

    # Zero-inflation: effect of normalized credit score
    # Better credit (higher normalized value) → slightly more likely structural zero
    gamma_credit: float = 0.12

    # Risk score coefficients (log relative risks)
    # Applied to log(mu) to shift expected claim frequency

    # Young driver: age < 25.
    beta_young: float = 0.40

    # Senior driver: age > 65
    beta_senior: float = 0.40

    # Male driver
    beta_male: float = 0.04

    # Single/unmarried
    beta_single: float = 0.10

    # Credit score: log RR per 100 points above/below 650
    # Negative: higher credit → lower frequency
    beta_credit_100pts: float = -0.28

    # Commercial vehicle use
    beta_commercial: float = 0.35

    # Farmer/artisan vehicle use. Calibrated: target GLM coef -0.808;
    # copula offset ≈ -0.277 → beta = -0.808 - (-0.277) = -0.531
    beta_farmer_artisan: float = -0.53

    # Vehicle age in years, centered at median (≈9). Negative: older cars → fewer claims.
    beta_car_age: float = -0.07

    # Log annual miles centered at log(12 000)
    # 1 unit = 1 log-unit increase in miles (roughly 2.7× more miles)
    beta_log_miles: float = 0.22

    # Annual pct driven above mean (0.33).
    # Calibrated to SBV Q5 log-RR ≈ 2.80: beta ≈ 2.80 / (1 - 0.33) = 4.2
    beta_pct_driven: float = 2.40

    # Hard braking: log1p(brake_9_miles), centred at log1p(SBV median=2) ≈ 1.0986
    beta_hard_brake: float = 0.25
    brake_9_center: float = 1.0986

    # Hard acceleration: log1p(accel_9_miles), centred at log1p(SBV median=0) = 0.0
    beta_hard_accel: float = 0.18
    accel_9_center: float = 0.0

    # Territory log-RR: linear function of numeric zone code (range 11–91).
    # delta(terr) = territory_beta * (terr - territory_center)
    territory_beta: float = 0.004
    territory_center: float = 50.0


@dataclass
class SeverityParams:
    """
    Parameters for the gamma claim severity model.

    Model structure:
        A_i | N_i = n, X_i ~ Gamma(shape = shape_per_claim * n,
                                    scale = scale_base * sev_factor(X_i))

    where sev_factor(X_i) is a multiplicative adjustment.

    This implies E[A | N=n, X] = n * shape_per_claim * scale_base * sev_factor(X).
    For the reference driver with N=1: E[A] = shape_per_claim * scale_base.
    """

    # Base expected severity for N=1 claim, reference driver:
    #   E[A] = shape_per_claim * scale_base
    shape_per_claim: float = 2.0
    scale_base: float = 1_567.0   # → E[A|N=1] = 2 * 1567 = 3 134 monetary units

    # Minimum claim threshold (hard floor after sampling)
    min_claim_amount: float = 1.0

    # Risk-factor adjustments on severity scale (log-linear)
    # Applied to scale_base: sev_factor = exp(sum of applicable coefficients)

    # Young driver
    beta_young: float = 0.20

    # Log annual miles centered at log(12 000)
    beta_log_miles: float = 0.12

    # Car age (centered at 7 years)
    beta_car_age: float = 0.018


# ---------------------------------------------------------------------------
# Assembled preset
# ---------------------------------------------------------------------------


@dataclass
class CorePreset:
    """
    Complete parameter set for SynthDrive v0.1 'core' preset.

    Instantiate with CorePreset() to get calibrated defaults.
    Individual sub-parameters can be overridden after construction.
    """

    exposure: ExposureParams = field(default_factory=ExposureParams)
    driver: DriverParams = field(default_factory=DriverParams)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    telematics: TelematicsParams = field(default_factory=TelematicsParams)
    copula: CopulaParams = field(default_factory=CopulaParams)
    frequency: FrequencyParams = field(default_factory=FrequencyParams)
    severity: SeverityParams = field(default_factory=SeverityParams)

    name: str = "core"
    description: str = (
        "Core preset for SynthDrive v0.1. "
        "Calibrated to produce a plausible personal auto telematics portfolio "
        "consistent with the structure and claim statistics of "
        "So, Boucher, and Valdez (2021). "
        "Operates in parameter mode; does not require a seed CSV."
    )


def load_core_preset() -> CorePreset:
    """Return a CorePreset with all calibrated default parameters."""
    return CorePreset()
