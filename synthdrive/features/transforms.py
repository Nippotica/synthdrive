"""
Marginal distribution specifications and transforms for SynthDrive v0.1.

Each variable has an associated marginal distribution.  The transforms module
provides functions that convert uniform samples from the Gaussian copula into
draws from each variable's marginal distribution.

For seed mode, empirical marginals are derived from the loaded CSV.
For parameter mode, parametric marginals are constructed from CorePreset values.

All proportion variables are on [0.0, 1.0].
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import (
    beta as beta_dist,
    gamma as gamma_dist,
    lognorm,
    norm,
    truncnorm,
)


# ---------------------------------------------------------------------------
# Marginal specification
# ---------------------------------------------------------------------------


class MarginalSpec(NamedTuple):
    """
    Specification for the marginal distribution of a single variable.

    ``ppf`` is the percent-point function (inverse CDF): given a uniform
    sample u in (0, 1), returns the corresponding variable value.
    ``post_process`` is an optional function applied after the ppf transform
    (e.g., rounding integers, clipping).
    """
    name: str
    ppf: Callable[[np.ndarray], np.ndarray]
    post_process: Optional[Callable[[np.ndarray], np.ndarray]] = None


# ---------------------------------------------------------------------------
# Parametric marginal constructors
# ---------------------------------------------------------------------------


def truncated_normal_ppf(
    mean: float,
    std: float,
    low: float,
    high: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the PPF of a truncated normal distribution."""
    a = (low - mean) / std
    b = (high - mean) / std
    dist = truncnorm(a=a, b=b, loc=mean, scale=std)
    return dist.ppf


def lognormal_ppf(
    log_mean: float,
    log_std: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return the PPF of a log-normal distribution.

    Parameters correspond to the mean and std of the underlying normal
    (i.e., log_mean = E[log X], log_std = Std[log X]).
    """
    dist = lognorm(s=log_std, scale=np.exp(log_mean))
    def ppf(u: np.ndarray) -> np.ndarray:
        x = dist.ppf(u)
        if low is not None:
            x = np.maximum(x, low)
        if high is not None:
            x = np.minimum(x, high)
        return x
    return ppf


def gamma_ppf(
    shape: float,
    scale: float,
    low: float = 0.0,
    high: Optional[float] = None,
    shift: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the PPF of a (optionally shifted) gamma distribution, clipped to [low, high]."""
    dist = gamma_dist(a=shape, scale=scale)
    def ppf(u: np.ndarray) -> np.ndarray:
        x = dist.ppf(u) + shift
        x = np.maximum(x, low)
        if high is not None:
            x = np.minimum(x, high)
        return x
    return ppf


def beta_ppf(
    alpha: float,
    beta: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the PPF of a Beta(alpha, beta) distribution."""
    dist = beta_dist(a=alpha, b=beta)
    return dist.ppf


def exponential_ppf(
    mean: float,
    low: float = 0.0,
    high: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the PPF of an exponential distribution."""
    from scipy.stats import expon
    dist = expon(scale=mean)
    def ppf(u: np.ndarray) -> np.ndarray:
        x = dist.ppf(u)
        x = np.maximum(x, low)
        if high is not None:
            x = np.minimum(x, high)
        return x
    return ppf


def empirical_ppf(
    data: np.ndarray,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return an empirical PPF (inverse CDF) from observed data using linear
    interpolation between order statistics.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    # Quantile levels corresponding to each sorted observation
    quantile_levels = (np.arange(1, n + 1) - 0.5) / n  # avoid 0 and 1

    from scipy.interpolate import interp1d

    interp = interp1d(
        quantile_levels,
        sorted_data,
        kind="linear",
        bounds_error=False,
        fill_value=(sorted_data[0], sorted_data[-1]),
    )

    def ppf(u: np.ndarray) -> np.ndarray:
        x = interp(u)
        if low is not None:
            x = np.maximum(x, low)
        if high is not None:
            x = np.minimum(x, high)
        return x

    return ppf


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def to_int(arr: np.ndarray) -> np.ndarray:
    """Round and cast to int64."""
    return np.round(arr).astype(np.int64)


def clip(low: Optional[float] = None, high: Optional[float] = None) -> Callable:
    def _clip(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, a_min=low, a_max=high)
    return _clip


def compose(*fns: Callable) -> Callable:
    """Compose multiple post-processing functions left-to-right."""
    def _composed(arr: np.ndarray) -> np.ndarray:
        for fn in fns:
            arr = fn(arr)
        return arr
    return _composed


# ---------------------------------------------------------------------------
# Build marginal specs from CorePreset (parameter mode)
# ---------------------------------------------------------------------------


def build_marginal_specs(params: object) -> dict[str, MarginalSpec]:
    """
    Build a dictionary of MarginalSpec objects from a CorePreset.

    These cover only the variables that flow through the Gaussian copula
    (the nine continuous variables in CopulaParams.copula_vars).  Categorical,
    compositional, and derived variables are handled separately in sampler.py.

    Parameters
    ----------
    params : CorePreset
        Full preset from synthdrive.presets.core.

    Returns
    -------
    dict mapping variable name → MarginalSpec
    """
    d_params = params.driver
    v_params = params.vehicle
    t_params = params.telematics

    specs: dict[str, MarginalSpec] = {}

    # --- insured_age: truncated normal → integer
    specs["insured_age"] = MarginalSpec(
        name="insured_age",
        ppf=truncated_normal_ppf(
            mean=d_params.age_mean,
            std=d_params.age_std,
            low=d_params.age_min,
            high=d_params.age_max,
        ),
        post_process=compose(
            clip(low=d_params.age_min, high=d_params.age_max),
            to_int,
        ),
    )

    # --- credit_score: truncated normal → integer
    specs["credit_score"] = MarginalSpec(
        name="credit_score",
        ppf=truncated_normal_ppf(
            mean=d_params.credit_mean,
            std=d_params.credit_std,
            low=d_params.credit_min,
            high=d_params.credit_max,
        ),
        post_process=compose(
            clip(low=d_params.credit_min, high=d_params.credit_max),
            to_int,
        ),
    )

    # --- years_no_claims: exponential → integer
    specs["years_no_claims"] = MarginalSpec(
        name="years_no_claims",
        ppf=exponential_ppf(
            mean=d_params.years_no_claims_mean,
            low=0.0,
            high=float(d_params.years_no_claims_max),
        ),
        post_process=compose(
            clip(low=0.0, high=float(d_params.years_no_claims_max)),
            to_int,
        ),
    )

    # --- car_age: shifted gamma → integer (shift = -2 for model-year convention)
    specs["car_age"] = MarginalSpec(
        name="car_age",
        ppf=gamma_ppf(
            shape=v_params.car_age_shape,
            scale=v_params.car_age_scale,
            low=float(v_params.car_age_min),
            high=float(v_params.car_age_max),
            shift=float(v_params.car_age_min),
        ),
        post_process=compose(
            clip(low=float(v_params.car_age_min), high=float(v_params.car_age_max)),
            to_int,
        ),
    )

    # --- annual_miles_drive: log-normal → float clipped at bounds
    specs["annual_miles_drive"] = MarginalSpec(
        name="annual_miles_drive",
        ppf=lognormal_ppf(
            log_mean=v_params.annual_miles_log_mean,
            log_std=v_params.annual_miles_log_std,
            low=500.0,
            high=60_000.0,
        ),
        post_process=clip(low=500.0, high=60_000.0),
    )

    # --- avg_days_week: truncated normal clipped to [0, 7]
    specs["avg_days_week"] = MarginalSpec(
        name="avg_days_week",
        ppf=truncated_normal_ppf(
            mean=t_params.avg_days_mean,
            std=t_params.avg_days_std,
            low=0.0,
            high=7.0,
        ),
        post_process=clip(low=0.0, high=7.0),
    )

    # --- annual_pct_driven: Beta distribution
    specs["annual_pct_driven"] = MarginalSpec(
        name="annual_pct_driven",
        ppf=beta_ppf(
            alpha=t_params.annual_pct_alpha,
            beta=t_params.annual_pct_beta,
        ),
        post_process=clip(low=0.0, high=1.0),
    )

    # --- accel_9_miles: log-normal → non-negative integer (raw event count)
    specs["accel_9_miles"] = MarginalSpec(
        name="accel_9_miles",
        ppf=lognormal_ppf(
            log_mean=t_params.accel_9_log_mean,
            log_std=t_params.accel_9_log_std,
            low=0.0,
        ),
        post_process=compose(clip(low=0.0), to_int),
    )

    # --- brake_9_miles: log-normal → non-negative integer (raw event count)
    specs["brake_9_miles"] = MarginalSpec(
        name="brake_9_miles",
        ppf=lognormal_ppf(
            log_mean=t_params.brake_9_log_mean,
            log_std=t_params.brake_9_log_std,
            low=0.0,
        ),
        post_process=compose(clip(low=0.0), to_int),
    )

    return specs


# ---------------------------------------------------------------------------
# Build marginal specs from seed data (seed mode)
# ---------------------------------------------------------------------------


def build_marginal_specs_from_seed(
    seed_df: pd.DataFrame,
    copula_vars: Sequence[str],
) -> dict[str, MarginalSpec]:
    """
    Build empirical marginal specs from the loaded seed DataFrame.

    For each variable in copula_vars, fits an empirical inverse CDF.
    Integer variables are rounded after transformation.

    Parameters
    ----------
    seed_df : pd.DataFrame
        The loaded seed CSV (So–Boucher–Valdez or similar).
    copula_vars : sequence of str
        Names of variables to include.

    Returns
    -------
    dict mapping variable name → MarginalSpec
    """
    INTEGER_VARS = {"insured_age", "credit_score", "years_no_claims", "car_age",
                    "accel_9_miles", "brake_9_miles"}

    specs: dict[str, MarginalSpec] = {}
    for var in copula_vars:
        if var not in seed_df.columns:
            continue
        data = seed_df[var].dropna().values.astype(float)
        if len(data) < 10:
            continue

        ppf_fn = empirical_ppf(data, low=data.min(), high=data.max())
        post = to_int if var in INTEGER_VARS else clip(low=float(data.min()))

        specs[var] = MarginalSpec(name=var, ppf=ppf_fn, post_process=post)

    return specs


# ---------------------------------------------------------------------------
# Apply marginal transforms to copula uniform samples
# ---------------------------------------------------------------------------


def apply_marginals(
    U: np.ndarray,
    copula_var_order: Sequence[str],
    specs: dict[str, MarginalSpec],
) -> dict[str, np.ndarray]:
    """
    Transform a (n, d) array of uniform copula samples into variable-scale arrays.

    Parameters
    ----------
    U : ndarray of shape (n, d)
        Uniform samples from the Gaussian copula.
    copula_var_order : sequence of str
        Variable names in the same column order as U.
    specs : dict
        MarginalSpec objects keyed by variable name.

    Returns
    -------
    dict mapping variable name → 1-D array of length n
    """
    result: dict[str, np.ndarray] = {}
    for i, var in enumerate(copula_var_order):
        if var not in specs:
            continue
        spec = specs[var]
        u_col = U[:, i]
        x = spec.ppf(u_col)
        if spec.post_process is not None:
            x = spec.post_process(x)
        result[var] = np.asarray(x)
    return result
