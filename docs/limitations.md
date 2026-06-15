# SynthDrive v0.1 — Known Limitations

## Modelling Limitations

**L-01: Gaussian copula tails.**
The Gaussian copula produces light joint tails.  Tail events (e.g. simultaneous
high mileage + extreme hard braking + high claim count) are underrepresented
relative to real telematics portfolios that may exhibit tail dependence.

**L-02: Parametric marginals.**
In parameter mode, continuous variables are modelled with simple parametric
families (TruncNormal, LogNormal, Beta, Gamma).  Real marginals often have
multi-modality, heavier tails, or point masses that these families cannot
capture.

**L-03: Independent categoricals.**
In parameter mode, categorical variables (sex, car_use, territory, region)
are sampled independently from their marginals.  Real portfolios exhibit
correlations between categoricals (e.g. commercial vehicles concentrate in
certain territories) that are not replicated.

**L-04: Simplified severity.**
The Gamma severity model does not distinguish claim types or apply a
development pattern.  Reported amounts are treated as ultimate settled amounts.

**L-05: No temporal structure.**
Telematics features are assumed i.i.d. across policies.  Seasonal driving
patterns, within-driver consistency across renewals, and trip-level data are
not modelled.

**L-06: Threshold sequence derivation.**
Accel/brake threshold sequences at non-anchor thresholds (6, 8, 10, 11, 12
mph/s) are derived from the anchor (9 mph/s) using fixed ratio tables.  The
ratios are plausible but not empirically calibrated.

**L-07: Rush-hour and late-night time bands are independent.**
In parameter mode, `pct_drive_rush_am`, `pct_drive_rush_pm`, and the
`pct_drive_Nhrs` variables are sampled independently from Beta distributions
and are not constrained to sum to ≤ 1 with the day-of-week proportions.

## Scope Limitations

**S-01: One preset.**
Only the `core` preset (generic North American personal auto) is available.
The Japan preset is a stub; other geographic presets are not yet implemented.

**S-02: No commercial lines.**
SynthDrive v0.1 targets personal auto policies only.  Commercial fleet, cargo,
and specialty lines are out of scope.

**S-03: No policy-level features beyond listed schema.**
Features such as deductible amount, coverage limits, and prior claim history
beyond `years_no_claims` are not included.

## Seed Mode Limitations

**S-04: Empirical marginal fitting quality.**
With N < 5 000 seed observations, empirical marginals and Spearman
correlations will be imprecise.  For robust seed-mode output, use the full
SBV dataset (~100 000 rows).

**S-05: Column name mapping may be incomplete.**
`load_seed.py` maps known SBV R-style column names.  Datasets from other
sources may require a custom column rename before passing to seed mode.

## Road Map

See `docs/roadmap_v02.md` for planned improvements in v0.2.
