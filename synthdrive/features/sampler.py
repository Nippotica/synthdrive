"""
Feature sampler for SynthDrive v0.1.

This module combines the Gaussian copula, marginal transforms, categorical
samplers, and derived-variable calculations into a single FeatureSampler class.

Sampling flow
-------------
1.  Sample continuous copula variables (insured_age, credit_score,
    years_no_claims, car_age, annual_miles_drive, avg_days_week,
    annual_pct_driven, accel_9_miles, brake_9_miles) via the Gaussian copula
    and variable-specific inverse CDFs.

2.  Sample categorical variables independently from their marginal proportions:
    insured_sex, marital_status, car_use, territory, region.

3.  Sample policy duration from a discrete mixture; set exposure = duration.

4.  Derive total_miles_driven from annual_miles_drive, exposure, and noise.

5.  Sample day-of-week proportions from a Dirichlet distribution.
    Compute derived pct_drive_wkday and pct_drive_wkend.

6.  Sample time-of-day variables (rush_am, rush_pm, 2hrs, 3hrs, 4hrs)
    independently from Beta distributions.

7.  Derive accel/brake threshold sequences from the copula samples using
    fixed ratio tables and small multiplicative noise.  Apply cumulative
    minimum to enforce monotonicity.

8.  Sample left and right turn intensity sequences.

9.  Assemble into a pandas DataFrame.

10. Apply constraint enforcement (via features.constraints).

Seed mode vs. parameter mode
-----------------------------
If a SeedData object is provided, the copula is fit from the seed data and
empirical marginals are used.  Otherwise, calibrated default parameters from
the CorePreset are used.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from synthdrive.data.schema import (
    ACCEL_COLUMNS,
    BRAKE_COLUMNS,
    DOW_COLUMNS,
    LEFT_TURN_COLUMNS,
    RIGHT_TURN_COLUMNS,
)
from synthdrive.features.constraints import enforce_all_constraints
from synthdrive.features.copula import GaussianCopula, copula_from_params, fit_copula_from_data
from synthdrive.features.transforms import (
    apply_marginals,
    build_marginal_specs,
    build_marginal_specs_from_seed,
)


class FeatureSampler:
    """
    Generates the feature portion (X) of a synthetic telematics portfolio.

    Parameters
    ----------
    params : CorePreset
        Calibrated parameters from synthdrive.presets.core.
    seed_data : optional SeedData
        If provided, the copula and marginals are fit from this data.
        If None, the preset parameters are used directly.
    """

    def __init__(self, params: object, seed_data: Optional[object] = None) -> None:
        self.params = params
        self.seed_data = seed_data

        # Build copula
        if seed_data is not None and hasattr(seed_data, "df"):
            seed_df = seed_data.df
            copula_vars = list(params.copula.copula_vars)
            available = [v for v in copula_vars if v in seed_df.columns]
            self.copula_vars_ = tuple(available)
            copula_data = seed_df[list(available)].dropna().values.astype(float)
            self.copula_ = fit_copula_from_data(copula_data, variable_names=self.copula_vars_)
            self.marginal_specs_ = build_marginal_specs_from_seed(seed_df, available)
        else:
            self.copula_vars_ = params.copula.copula_vars
            self.copula_ = copula_from_params(params.copula)
            self.marginal_specs_ = build_marginal_specs(params)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """
        Generate n rows of feature variables.

        Parameters
        ----------
        n : int
            Number of policies to generate.
        rng : numpy.random.Generator
            Seeded random number generator.

        Returns
        -------
        pd.DataFrame with one row per policy, all feature columns except
        claim variables (those are added by the caller).
        """
        cols: dict[str, np.ndarray] = {}

        # 1. Copula variables
        copula_cols = self._sample_copula_vars(n, rng)
        cols.update(copula_cols)

        # 2. Categorical variables
        cat_cols = self._sample_categoricals(n, rng)
        cols.update(cat_cols)

        # 3. Exposure / duration
        exposure_cols = self._sample_exposure(n, rng)
        cols.update(exposure_cols)

        # 4. Derived mileage
        cols["total_miles_driven"] = self._derive_total_miles(
            cols["annual_miles_drive"],
            cols["exposure"],
            cols["annual_pct_driven"],
            rng,
        )

        # 5. Day-of-week proportions
        dow_cols = self._sample_dow(n, rng)
        cols.update(dow_cols)

        # 6. Time-of-day variables
        tod_cols = self._sample_time_of_day(n, rng)
        cols.update(tod_cols)

        # 7. Accel / brake threshold sequences
        accel_cols = self._derive_event_sequence(
            base=cols["accel_9_miles"],
            ratios=self.params.telematics.accel_ratios,
            column_order=ACCEL_COLUMNS,
            noise_std=self.params.telematics.threshold_noise_std,
            rng=rng,
        )
        cols.update(accel_cols)

        brake_cols = self._derive_event_sequence(
            base=cols["brake_9_miles"],
            ratios=self.params.telematics.brake_ratios,
            column_order=BRAKE_COLUMNS,
            noise_std=self.params.telematics.threshold_noise_std,
            rng=rng,
        )
        cols.update(brake_cols)

        # 8. Turn intensity sequences
        turn_cols = self._sample_turn_intensities(n, rng)
        cols.update(turn_cols)

        # 9. Assemble DataFrame
        df = self._assemble_dataframe(cols, n)

        # 10. Constraint enforcement (non-claim constraints)
        df = enforce_all_constraints(df)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_copula_vars(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Sample continuous variables via the Gaussian copula."""
        U = self.copula_.sample_uniform(n=n, rng=rng)
        return apply_marginals(U, self.copula_vars_, self.marginal_specs_)

    def _sample_categoricals(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Sample categorical variables from marginal proportions."""
        v_params = self.params.vehicle
        d_params = self.params.driver

        cols: dict[str, np.ndarray] = {}

        # insured_sex: "Male" / "Female"
        cols["insured_sex"] = np.where(
            rng.random(n) < d_params.p_male, "Male", "Female"
        )

        # marital_status: "Married" / "Single"
        cols["marital_status"] = np.where(
            rng.random(n) < d_params.p_married, "Married", "Single"
        )

        # car_use: "Commute", "Private", "Commercial", "Farmer"
        car_use_cats = list(v_params.car_use_probs.keys())
        car_use_probs = np.array(list(v_params.car_use_probs.values()), dtype=float)
        car_use_probs /= car_use_probs.sum()  # normalize for safety
        car_use_idx = rng.choice(len(car_use_cats), size=n, p=car_use_probs)
        cols["car_use"] = np.array(car_use_cats)[car_use_idx]

        # territory: continuous integer zone code, range 11–91
        cols["territory"] = rng.integers(
            v_params.territory_min, v_params.territory_max + 1, size=n
        )

        # region: "Urban" / "Rural"
        reg_cats = list(v_params.region_probs.keys())
        reg_probs = np.array(list(v_params.region_probs.values()), dtype=float)
        reg_probs /= reg_probs.sum()
        reg_idx = rng.choice(len(reg_cats), size=n, p=reg_probs)
        cols["region"] = np.array(reg_cats)[reg_idx]

        return cols

    def _sample_exposure(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Sample policy duration (days) from a discrete mixture; exposure = duration / 365."""
        e_params = self.params.exposure
        values = np.array(e_params.duration_values, dtype=float)
        probs = np.array(e_params.duration_probs, dtype=float)
        probs /= probs.sum()
        idx = rng.choice(len(values), size=n, p=probs)
        duration = values[idx]
        exposure = duration / 365.0
        return {"duration": duration, "exposure": exposure}

    def _derive_total_miles(
        self,
        annual_miles: np.ndarray,
        exposure: np.ndarray,
        annual_pct: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Derive total_miles_driven from annual mileage, exposure, and annual pct driven.

        total ≈ annual_miles * exposure * (0.6 + 0.8 * annual_pct) * exp(noise)

        The factor (0.6 + 0.8 * annual_pct) spans 0.6 to 1.4, centred around 1
        when annual_pct ≈ 0.5.  This creates a plausible relationship without
        a perfect deterministic mapping.
        """
        # Multiplicative noise: exp(N(0, 0.12))
        noise = np.exp(rng.normal(0.0, 0.12, size=len(annual_miles)))
        factor = 0.60 + 0.80 * annual_pct
        total = annual_miles * exposure * factor * noise
        return np.maximum(total, 0.0)

    def _sample_dow(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Sample day-of-week proportions from a Dirichlet distribution.
        Normalize so they sum to 1; compute derived wkday/wkend aggregates.
        """
        t_params = self.params.telematics
        alpha = np.array(t_params.weekday_dirichlet, dtype=float)  # length 7
        dow_arr = rng.dirichlet(alpha, size=n)  # shape (n, 7)

        cols: dict[str, np.ndarray] = {}
        for i, day in enumerate(DOW_COLUMNS):
            cols[day] = dow_arr[:, i]

        cols["pct_drive_wkday"] = dow_arr[:, :5].sum(axis=1)
        cols["pct_drive_wkend"] = dow_arr[:, 5:].sum(axis=1)

        return cols

    def _sample_time_of_day(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Sample time-of-day proportion variables independently from Beta distributions.
        These are not constrained to sum to 1 — they represent fractions of driving
        in specific time windows, not an exhaustive partition.
        """
        t_params = self.params.telematics
        from scipy.stats import beta as beta_dist

        def _sample_beta(alpha: float, b: float) -> np.ndarray:
            return beta_dist.rvs(a=alpha, b=b, size=n, random_state=rng)

        cols: dict[str, np.ndarray] = {
            "pct_drive_rush_am": _sample_beta(t_params.rush_am_alpha, t_params.rush_am_beta),
            "pct_drive_rush_pm": _sample_beta(t_params.rush_pm_alpha, t_params.rush_pm_beta),
            "pct_drive_2hrs":    _sample_beta(t_params.night_2hrs_alpha, t_params.night_2hrs_beta),
            "pct_drive_3hrs":    _sample_beta(t_params.night_3hrs_alpha, t_params.night_3hrs_beta),
            "pct_drive_4hrs":    _sample_beta(t_params.night_4hrs_alpha, t_params.night_4hrs_beta),
        }
        # Clip to [0, 1]
        for key in cols:
            cols[key] = np.clip(cols[key], 0.0, 1.0)

        return cols

    def _derive_event_sequence(
        self,
        base: np.ndarray,
        ratios: dict,
        column_order: tuple,
        noise_std: float,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """
        Derive a monotone-decreasing threshold sequence from a base variable.

        Each column value = base * ratio * exp(N(0, noise_std)).
        After applying ratios, a cumulative minimum is taken left-to-right
        to enforce strict non-increasing order.

        Parameters
        ----------
        base : ndarray
            Anchor column values (e.g., accel_9_miles or brake_9_miles).
        ratios : dict
            Mapping from column name to the ratio relative to the anchor.
        column_order : tuple
            Column names in decreasing-threshold (non-increasing value) order.
        noise_std : float
            Standard deviation of the multiplicative log-normal noise.
        rng : numpy.random.Generator
        """
        n = len(base)
        present = [c for c in column_order if c in ratios]
        arr = np.zeros((n, len(present)), dtype=float)

        for i, col in enumerate(present):
            noise = np.exp(rng.normal(0.0, noise_std, size=n))
            arr[:, i] = np.maximum(np.round(base * ratios[col] * noise), 0.0)

        # Cumulative minimum enforces non-increasing order
        arr = np.minimum.accumulate(arr, axis=1)

        return {col: arr[:, i] for i, col in enumerate(present)}

    def _sample_turn_intensities(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Sample left and right turn intensity sequences.

        The base (threshold 0.8) is log-normally distributed.  Higher thresholds
        (0.9, 1.0, 1.1, 1.2) are derived by applying step ratios and noise, then
        cumulatively capped.

        Left and right turn intensities are correlated via a bivariate normal
        copula on the log scale.
        """
        t_params = self.params.telematics
        ratios = t_params.turn_intensity_step_ratios  # length 5
        noise_std = t_params.turn_intensity_noise_std

        # Sample correlated log-scale base values for left and right
        rho = t_params.left_right_turn_spearman
        # Convert Spearman to Pearson (approx bivariate normal)
        rho_p = 2.0 * np.sin(np.pi * rho / 6.0)
        cov = np.array([[1.0, rho_p], [rho_p, 1.0]])

        z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
        from scipy.stats import norm as norm_dist

        # Transform to log-space values
        u_left = norm_dist.cdf(z[:, 0])
        u_right = norm_dist.cdf(z[:, 1])

        # Apply log-normal marginals for the 0.8 threshold base
        from scipy.stats import lognorm
        left_dist = lognorm(
            s=t_params.left_08_log_std,
            scale=np.exp(t_params.left_08_log_mean),
        )
        right_dist = lognorm(
            s=t_params.right_08_log_std,
            scale=np.exp(t_params.right_08_log_mean),
        )

        left_base = np.maximum(left_dist.ppf(u_left), 0.0)
        right_base = np.maximum(right_dist.ppf(u_right), 0.0)

        def _build_sequence(
            base: np.ndarray,
            col_names: tuple,
        ) -> dict[str, np.ndarray]:
            k = len(col_names)
            arr = np.zeros((n, k), dtype=float)
            for i, ratio in enumerate(ratios[:k]):
                noise = np.exp(rng.normal(0.0, noise_std, size=n))
                arr[:, i] = np.maximum(np.round(base * ratio * noise), 0.0)
            arr = np.minimum.accumulate(arr, axis=1)
            return {col: arr[:, i] for i, col in enumerate(col_names)}

        cols: dict[str, np.ndarray] = {}
        cols.update(_build_sequence(left_base, LEFT_TURN_COLUMNS))
        cols.update(_build_sequence(right_base, RIGHT_TURN_COLUMNS))
        return cols

    def _assemble_dataframe(
        self,
        cols: dict[str, np.ndarray],
        n: int,
    ) -> pd.DataFrame:
        """
        Assemble sampled arrays into a DataFrame with correct dtypes.

        Categorical columns are stored as pandas Categorical for memory
        efficiency.  Claim columns are absent at this stage.
        """
        df = pd.DataFrame(cols, index=range(n))

        # Cast integer columns
        for int_col in ("insured_age", "credit_score", "years_no_claims", "car_age",
                        "territory"):
            if int_col in df.columns:
                df[int_col] = df[int_col].astype(np.int64)

        # Cast categorical columns
        cat_map = {
            "insured_sex":    ("Male", "Female"),
            "marital_status": ("Married", "Single"),
            "car_use":        ("Commute", "Private", "Commercial", "Farmer"),
            "region":         ("Urban", "Rural"),
        }
        for col, cats in cat_map.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str), categories=cats)

        # Ensure float64 for all remaining float columns
        float_cols = [
            c for c in df.columns
            if c not in ("insured_sex", "marital_status", "car_use", "region",
                         "insured_age", "credit_score", "years_no_claims", "car_age",
                         "territory")
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        return df
