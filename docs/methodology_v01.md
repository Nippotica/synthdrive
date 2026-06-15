# SynthDrive v0.1 — Methodology

> **Status:** v0.1 release candidate  
> **Formal spec:** `docs/synthdrive_formal_v2.pdf`  
> **Last revised:** corrected spec (v2)

---

## 1. Purpose and scope

SynthDrive v0.1 generates actuarially structured synthetic telematics datasets for motor
insurance research. Its primary use cases are usage-based insurance (UBI) pricing research,
claim frequency and severity modeling, fraud model testing, and synthetic-data benchmarking.

The v0.1 engine uses classical statistical and actuarial methods: a Gaussian copula for the
feature joint distribution, a zero-inflated negative binomial (ZINB) model for claim counts,
and a Gamma severity model for aggregate claim amounts. It runs on a standard laptop
without GPU hardware and produces fully reproducible output.

SynthDrive v0.1 does **not** use deep generative models. TabSyn and TabDDPM are planned
for v0.2.

---

## 2. Data lineage and intellectual honesty

SynthDrive v0.1 can be initialized from the public synthetic telematics dataset of
So, Boucher, and Valdez (2021), hereafter SBV.

The SBV dataset is itself synthetic, generated from a private Canadian insurer's telematics
portfolio using extended SMOTE and feedforward neural networks. SynthDrive trained on the
SBV CSV is therefore a synthetic generator trained from a public synthetic telematics dataset,
not from original insurer data.

Correct description of the data lineage:

> SynthDrive Core can be initialized from the public So–Boucher–Valdez synthetic
> telematics dataset.

Incorrect description that must not be used:

> SynthDrive learns directly from real insurer telematics data.

---

## 3. Dataset schema

The generated dataset has policy-level rows. Variable types follow Table 4 of SBV (2021):
categorical, continuous/integer, percentage, and compositional.

### 3.1 Policy and exposure variables

| Column     | Type    | Range / Values             | Description                            |
|------------|---------|----------------------------|----------------------------------------|
| `policy_id`| string  | unique                     | Policy identifier                      |
| `duration` | integer | [22, 366] days             | Coverage period in **days**            |
| `exposure` | float   | (0, 1]                     | `duration / 365`; fractional year      |

Duration is measured in days. Exposure is the fractional policy year:

```
exposure = duration / 365
```

Exposure is clipped to the interval (0, 1].

### 3.2 Driver variables

| Column           | Type    | Range / Values                  | Description                       |
|------------------|---------|---------------------------------|-----------------------------------|
| `insured_age`    | integer | [16, 103]                       | Age of insured driver, years      |
| `insured_sex`    | string  | `"Male"`, `"Female"`            | Sex of insured driver             |
| `marital_status` | string  | `"Married"`, `"Single"`         | Marital status                    |
| `credit_score`   | float   | continuous                      | Credit score of insured driver    |
| `years_no_claims`| integer | [0, 79], < `insured_age`        | Years without any claim           |

Categorical variables `insured_sex` and `marital_status` are stored as plain strings
(`"Male"`, `"Female"`, `"Married"`, `"Single"`). They are not encoded as integers or
one-hot vectors in the output CSV.

### 3.3 Vehicle variables

| Column               | Type    | Range / Values                                 | Description                         |
|----------------------|---------|------------------------------------------------|-------------------------------------|
| `car_age`            | integer | [−2, 20]                                       | Age of vehicle in years             |
| `car_use`            | string  | `"Commute"`, `"Private"`, `"Commercial"`, `"Farmer"` | Use of vehicle             |
| `territory`          | integer | integer zone codes in {11, 12, …, 91}          | Territorial location code (55 zones)|
| `region`             | string  | `"Urban"`, `"Rural"`                           | Regional type                       |
| `annual_miles_drive` | float   | > 0, miles                                     | Annual miles declared by driver     |

**Territory** is an integer zone code in the range [11, 91] with 55 distinct values.
It is not a six-level categorical; it is a numeric code used as a risk classification variable.

**Car age** can be negative (range [−2, 20]) because some policies cover vehicles purchased
up to two years before the model year.

**Car use** is stored as a plain string: `"Commute"`, `"Private"`, `"Commercial"`, or
`"Farmer"`. Encoding is not applied in the output CSV.

**Mileage** is in **miles** throughout v0.1, matching the SBV column labels
(`Annual.miles.drive`, `Total.miles.driven`). Kilometres are planned for the Japan preset
in v0.2.

### 3.4 Telematics variables

| Column                  | Type    | Description                                        |
|-------------------------|---------|----------------------------------------------------|
| `annual_pct_driven`     | float   | Days vehicle used / 365, range [0, 1]              |
| `total_miles_driven`    | float   | Total distance driven in miles, ≥ 0                |
| `avg_days_week`         | float   | Mean number of days used per week                  |
| `pct_drive_mon` … `pct_drive_sun` | float | % of driving on each day of the week; sum = 100 |
| `pct_drive_wkday`       | float   | % of driving on weekdays (Mon–Fri)                 |
| `pct_drive_wkend`       | float   | % of driving on weekends (Sat–Sun)                 |
| `pct_drive_2hrs`        | float   | % of vehicle use within 2 hrs                      |
| `pct_drive_3hrs`        | float   | % of vehicle use within 3 hrs                      |
| `pct_drive_4hrs`        | float   | % of vehicle use within 4 hrs                      |
| `pct_drive_rush_am`     | float   | % of driving during AM rush hour                   |
| `pct_drive_rush_pm`     | float   | % of driving during PM rush hour                   |

#### Acceleration and braking columns

Acceleration and braking variables are **raw event counts**, not rates.
The column names embed a threshold in miles per second squared:

| Threshold set | Columns                                                    |
|---------------|------------------------------------------------------------|
| {6, 8, 9, 11, 12, 14} mph/s | `accel_6_miles`, `accel_8_miles`, `accel_9_miles`, `accel_11_miles`, `accel_12_miles`, `accel_14_miles` |
| {6, 8, 9, 11, 12, 14} mph/s | `brake_6_miles`, `brake_8_miles`, `brake_9_miles`, `brake_11_miles`, `brake_12_miles`, `brake_14_miles` |

These are counts of sudden-acceleration and hard-braking events observed during the
policy period. They are **not** normalized per 1,000 miles in the SynthDrive output,
even though the SBV paper describes them as "per 1,000 miles." The corrected v2 spec
treats them as raw counts; the SBV empirical means are used to calibrate the ratios
between threshold levels.

The threshold set is {6, 8, 9, 11, 12, 14}. Note that 10 is absent from the threshold
set.

#### Turn intensity columns

Turn intensity variables are **raw event counts** (not proportions):

```
left_turn_intensity_08, left_turn_intensity_09, left_turn_intensity_10,
left_turn_intensity_11, left_turn_intensity_12
right_turn_intensity_08, right_turn_intensity_09, right_turn_intensity_10,
right_turn_intensity_11, right_turn_intensity_12
```

The integer suffix is the intensity threshold (08 through 12).

### 3.5 Claim variables

| Column         | Type  | Constraints              | Description                         |
|----------------|-------|--------------------------|-------------------------------------|
| `claim_count`  | int   | ∈ {0, 1, 2, 3}          | Number of claims during observation |
| `claim_amount` | float | ≥ 0; = 0 if count = 0   | Aggregate claim amount              |
| `pure_premium` | float | ≥ 0                     | `claim_amount / exposure`           |

`claim_amount` is exactly zero whenever `claim_count` is zero.
This constraint is enforced programmatically after generation.

---

## 4. Enforced constraints

The generator enforces the following constraints after sampling. All violations are
corrected by clipping, normalization, or recalculation; they are not silently dropped.

### 4.1 Scalar bounds

```
0 < exposure ≤ 1
0 ≤ annual_pct_driven ≤ 1
total_miles_driven ≥ 0
claim_count ∈ {0, 1, 2, 3}
claim_amount ≥ 0
claim_amount = 0  if  claim_count = 0
insured_age ≥ 16
car_age ∈ [−2, 20]
years_no_claims < insured_age
```

### 4.2 Compositional constraints

Day-of-week percentages sum to 100%:

```
pct_drive_mon + pct_drive_tue + pct_drive_wed + pct_drive_thu
+ pct_drive_fri + pct_drive_sat + pct_drive_sun = 100
```

Weekday and weekend aggregates are derived, not independently sampled:

```
pct_drive_wkday = pct_drive_mon + pct_drive_tue + pct_drive_wed
                + pct_drive_thu + pct_drive_fri
pct_drive_wkend = pct_drive_sat + pct_drive_sun
```

Percentages are stored in [0, 100], not [0, 1]. This is consistent with the SBV
column scale.

### 4.3 Integer types

`claim_count`, `insured_age`, `car_age`, `years_no_claims`, `territory`,
`duration`, and all `accel_*`, `brake_*`, `left_turn_intensity_*`, and
`right_turn_intensity_*` columns are stored as integers.

---

## 5. Feature generation model

### 5.1 Gaussian copula

The 9 continuous/integer features listed in Table 5.1 are generated jointly from a
Gaussian copula fit to their rank correlations in the SBV seed data.

| Variable              | Marginal distribution |
|-----------------------|-----------------------|
| `insured_age`         | empirical             |
| `credit_score`        | empirical             |
| `annual_miles_drive`  | empirical             |
| `annual_pct_driven`   | Beta                  |
| `total_miles_driven`  | empirical             |
| `avg_days_week`       | empirical             |
| `car_age`             | empirical             |
| `years_no_claims`     | empirical             |
| `duration`            | empirical             |

Each marginal is fitted independently; the copula captures the rank-based dependence
structure. After sampling from the copula, the marginals are transformed back via
inverse CDF (quantile transform).

### 5.2 Categorical variables

Categorical variables (`insured_sex`, `marital_status`, `car_use`, `region`) are
sampled independently from their empirical proportions in the seed data. They are
not modeled jointly with the continuous variables in v0.1.

### 5.3 Compositional telematics variables

Day-of-week percentages are sampled from a Dirichlet distribution whose concentration
parameters are calibrated from the SBV seed data. After sampling, the simplex is
rescaled to sum to 100%.

`pct_drive_wkday` and `pct_drive_wkend` are computed from the day-of-week columns
after sampling, not independently drawn.

### 5.4 Acceleration and braking ratios

The accel/brake columns are generated by sampling a base count from the SBV empirical
distribution, then scaling to each threshold level using ratios calibrated from SBV
empirical means. Let `μ_t` denote the SBV mean count at threshold `t`. The ratio for
threshold `t` relative to the baseline threshold 6 is:

```
ratio_t = μ_t / μ_6     for t ∈ {6, 8, 9, 11, 12, 14}
```

Turn intensity columns are generated similarly using threshold-level ratios calibrated
from the SBV means for thresholds 08–12.

---

## 6. Claim frequency model

### 6.1 Model specification

Claim counts are generated from a zero-inflated negative binomial (ZINB) model:

```
N_i ~ ZINB(μ_i, α, π_i)
```

where:
- `N_i` is the claim count for policy `i`
- `μ_i` is the expected count (before zero-inflation)
- `α` is the overdispersion parameter (common across policies in v0.1)
- `π_i` is the structural zero probability

The log of the expected count includes an exposure offset:

```
log(μ_i) = log(exposure_i) + log(base_rate) + f(X_i)
```

### 6.2 Calibration to SBV

The parameters `α`, `base_rate`, and the risk score function `f(X_i)` are calibrated
so that the generated portfolio matches the SBV claim frequency distribution:

- Zero-claim rate: approximately 95.7%
- One-claim rate: approximately 4.1%
- Two-claim rate: approximately 0.2%
- Three-claim rate: approximately 0.01%
- Overall claim frequency: approximately 0.0523 claims per unit exposure

The frequency sanity check uses a ceiling of 0.50 claims per unit exposure. This is
a validation threshold, not a model parameter.

#### Calibrated parameter values

| Parameter | Value | Description |
|---|---|---|
| `base_rate` | 0.0165 | Baseline annual claim rate for the reference driver |
| `alpha` | 0.30 | NB overdispersion: Var[N] = μ + 0.30·μ² |
| `gamma_0` | −1.735 | Zero-inflation logit intercept (≈15% structural zeros at reference) |
| `gamma_ynclaims` | 0.035 | Zero-inflation: effect of years with no claims |
| `gamma_credit` | 0.12 | Zero-inflation: effect of normalised credit score |

`base_rate` and `beta_young` were jointly calibrated via grid search to satisfy two
constraints simultaneously: overall frequency ≈ 0.0523 and 16–25 age-band frequency
≈ 0.085.

### 6.3 Risk factors in f(X_i)

The additive log-RR risk score is:

```
f(X_i) = beta_young  · 1(age < 25)
        + beta_senior · 1(age > 65)
        + beta_male   · 1(sex = Male)
        + beta_single · 1(marital = Single)
        + beta_credit_100pts · (credit_score − 650) / 100
        + beta_commercial    · 1(car_use = Commercial)
        + beta_farmer        · 1(car_use = Farmer)
        + beta_car_age       · (car_age − 9)
        + beta_log_miles     · (log(miles) − log(12 000))
        + beta_pct_driven    · (annual_pct_driven − 0.33)
        + beta_hard_brake    · (log1p(brake_9_miles) − 1.0986)
        + beta_hard_accel    · (log1p(accel_9_miles) − 0.0)
        + territory_beta     · (territory − 50)
```

#### Brake and acceleration transformation

`brake_9_miles` and `accel_9_miles` are **log1p-transformed** before entering the
risk score. Without this transformation, raw event counts with extreme outliers (up to
800+) produce unbounded risk scores. The centering constants are calibrated from
SBV empirical medians of the corresponding columns (`Brake.09miles` median = 2,
`Accel.09miles` median = 0), giving:

```
brake term = beta_hard_brake · (log1p(brake_9_miles) − log1p(2)) = … · (log1p(brake_9_miles) − 1.0986)
accel term = beta_hard_accel · (log1p(accel_9_miles) − log1p(0)) = … · log1p(accel_9_miles)
```

#### Calibrated coefficient values

| Parameter | Value | Notes |
|---|---|---|
| `beta_young` | −0.28 | Age < 25; negative offsets copula-driven feature excess for young drivers |
| `beta_senior` | 0.40 | Age > 65 |
| `beta_male` | 0.04 | Male driver |
| `beta_single` | 0.10 | Single/unmarried |
| `beta_credit_100pts` | −0.28 | Per 100 points above/below 650 |
| `beta_commercial` | 0.35 | Commercial vehicle use |
| `beta_farmer_artisan` | −0.53 | Farmer/artisan; calibrated to SBV GLM coef ≈ −0.81 |
| `beta_car_age` | −0.07 | Per year above median (9 yrs); older cars → fewer claims |
| `beta_log_miles` | 0.22 | Log miles centred at log(12 000) |
| `beta_pct_driven` | 4.2 | Calibrated to SBV Q5 log-RR ≈ 2.80 |
| `beta_hard_brake` | 0.25 | log1p(brake_9_miles) centred at 1.0986 |
| `beta_hard_accel` | 0.18 | log1p(accel_9_miles) centred at 0.0 |
| `territory_beta` | 0.004 | Linear on zone code, centred at 50 |

The coefficients are calibrated to recover approximately the same Poisson GLM
relativities as the SBV data. See `output/glm_comparison.csv` for the validation
comparison.

---

## 7. Claim severity model

### 7.1 Model specification

Aggregate claim amount is modeled only for policies with `claim_count > 0`:

```
A_i | N_i = n, X_i ~ Gamma(shape = shape_per_claim × n, scale = scale_base × sev_factor(X_i))
```

where `sev_factor(X_i) = exp(sum of applicable severity log-RR coefficients)`. For the
reference driver with a single claim, `E[A] = shape_per_claim × scale_base`.

After sampling, a minimum claim floor of `min_claim_amount` is enforced for all
policies with `N_i > 0`. In v0.1, the severity model uses a single Gamma component;
a mixture model is planned for v0.2.

### 7.2 Calibrated parameter values

| Parameter | Value | Description |
|---|---|---|
| `shape_per_claim` | 2.0 | Gamma shape per claim; scales with claim count |
| `scale_base` | 1,567 | Gamma scale for reference driver; E[A\|N=1] = 2 × 1,567 = $3,134 |
| `min_claim_amount` | 1.0 | Hard floor (monetary units) for positive-claim policies |

### 7.3 Calibration targets

Calibration targets from SBV (positive claims only, NB_Claim = 1 subset):

| Statistic | SBV value | SynthDrive (n=100k) |
|-----------|-----------|---------------------|
| Mean      | ~$3,561   | ~$3,561             |
| Median    | ~$2,191   | ~$2,808             |
| p95       | ~$13,000  | ~$12,165            |

The `scale_base` was set so that the realised mean severity for positive-claim
policies matches the SBV mean of $3,561 exactly at n = 100,000 (random_state = 42).

---

## 8. Pure premium

Pure premium is computed after generation:

```
pure_premium_i = claim_amount_i / exposure_i
```

Pure premium is zero when `claim_amount` is zero.

---

## 9. Parameter mode and seed mode

SynthDrive v0.1 supports two generation modes.

**Parameter mode** uses hard-coded default parameters derived from the SBV calibration.
No CSV is required:

```python
df = generate(n=100_000, preset="core", random_state=42)
```

**Seed mode** reads the SBV CSV and fits the copula, marginals, and calibration
parameters from the data:

```python
df = generate(n=100_000, preset="core", seed_path="data/raw/telematics_syn-032021.csv",
              random_state=42)
```

Seed mode produces results that more closely match the SBV distributions because
the marginals are fitted directly from the data rather than from stored defaults.
Parameter mode is suitable when the SBV CSV is unavailable.

---

## 10. Validation methodology

The primary validation test is a GLM relativity comparison:

1. Fit a Poisson GLM on the SBV seed data.
2. Fit the same Poisson GLM on a SynthDrive-generated dataset of equal size.
3. Compare the log-RR coefficients term by term.

The formula used is:

```
claim_count ~ offset(log(exposure)) + insured_age_band + insured_sex
            + marital_status + car_use + credit_score_band
            + annual_miles_band + annual_pct_driven_band
            + territory_band + car_age_band
```

Continuous variables are banded into 5 quantile groups before fitting.
Results are saved in `output/glm_comparison.csv`.

Secondary validation includes marginal distribution comparisons, Spearman rank
correlation heatmaps, and frequency and severity plots. See `examples/validation_report.py`
and `output/figures/`.

The validation report makes explicit what passed, what failed, and what is approximate.
It does not assert that SynthDrive reproduces the original Canadian insurer data; the
comparison is against the SBV public synthetic dataset.

### 10.1 Validation results (n = 100,000, random-state = 42)

Both Poisson GLMs converged with 29 terms. Across all terms: mean absolute % difference
309%, median 41%. The mean is dominated by near-zero SBV coefficients in
`insured_age_band` Q3–Q4; the median is the more representative indicator of typical
agreement.

**Coefficient comparison — Poisson GLM log-RR**

| Variable | Band | SBV | SynthDrive | % diff | Status |
|---|---|---:|---:|---:|---|
| Intercept | — | −4.160 | −4.533 | −9.0 | calibrated |
| `annual_pct_driven_band` | Q2 | 1.363 | 1.176 | −13.7 | calibrated |
| `annual_pct_driven_band` | Q3 | 1.901 | 1.807 | −4.9 | calibrated |
| `annual_pct_driven_band` | Q4 | 2.352 | 2.759 | +17.3 | calibrated |
| `annual_pct_driven_band` | Q5 | 2.796 | 3.914 | +40.0 | moderate |
| `credit_score_band` | Q2 | −0.320 | −0.515 | −60.8 | moderate |
| `credit_score_band` | Q3 | −0.692 | −0.601 | +13.1 | calibrated |
| `credit_score_band` | Q4 | −0.786 | −0.721 | +8.3 | calibrated |
| `credit_score_band` | Q5 | −0.839 | −0.789 | +5.9 | calibrated |
| `car_age_band` | Q2 | −0.127 | −0.175 | −37.6 | moderate |
| `car_age_band` | Q3 | −0.324 | −0.291 | +10.4 | calibrated |
| `car_age_band` | Q4 | −0.416 | −0.434 | −4.5 | calibrated |
| `car_age_band` | Q5 | −0.699 | −0.585 | +16.3 | calibrated |
| `annual_miles_band` | Q2 | 0.038 | 0.059 | +57.0 | moderate |
| `annual_miles_band` | Q3 | 0.203 | 0.120 | −41.1 | moderate |
| `annual_miles_band` | Q4 | 0.215 | 0.221 | +2.6 | calibrated |
| `car_use` | Commute | −0.231 | −0.487 | −110.7 | limitation |
| `car_use` | Farmer | −0.808 | −1.095 | −35.5 | moderate |
| `car_use` | Private | −0.238 | −0.497 | −109.1 | limitation |
| `insured_age_band` | Q2 | −0.092 | −0.233 | −151.9 | limitation |
| `insured_age_band` | Q3 | +0.024 | −0.282 | −1291 ‡ | limitation |
| `insured_age_band` | Q4 | −0.007 | −0.363 | −5072 ‡ | limitation |
| `insured_age_band` | Q5 | −0.048 | −0.273 | −463 ‡ | limitation |
| `insured_sex` | Male | +0.037 | +0.082 | +119.0 | limitation |
| `marital_status` | Single | +0.046 | +0.086 | +85.9 | limitation |
| `territory_band` | Q2 | −0.085 | +0.049 | +157.2 | limitation |
| `territory_band` | Q3 | +0.038 | +0.018 | −54.2 | moderate |
| `territory_band` | Q4 | −0.017 | +0.111 | +759 ‡ | limitation |
| `territory_band` | Q5 | −0.179 | +0.197 | +210.2 | limitation |

‡ % diff is inflated by a near-zero SBV coefficient (|coef| < 0.05); the absolute
difference is the more informative measure in these cases.

Status key: *calibrated* = |% diff| ≤ 20%; *moderate* = 20–70%, same sign;
*limitation* = > 70% divergence or sign change.

**Well-calibrated variables.** `annual_pct_driven_band` (Q2–Q4), `car_age_band`
(Q3–Q5), `credit_score_band` (Q3–Q5), and `annual_miles_band` (Q4) are within ±20%
of SBV values. The intercept agrees to within 9%, confirming that the overall claim
frequency level is reproduced. These variables represent the primary pricing relativities
in telematics UBI models and their ordering is preserved.

**Documented limitations.** Four variables show material divergence.

- **`insured_age_band`:** SBV relativities for middle-age bands (Q3–Q4) are near zero
  (±0.024 log-RR), while SynthDrive produces negative values of −0.28 to −0.36. The
  copula-derived age distribution does not reproduce the near-flat SBV age–claim
  relationship in those bands. The extreme % diffs (up to −5072%) are an arithmetic
  artifact of near-zero denominators; the absolute deviations (0.28–0.36 log-RR) are
  the operative measure.

- **`territory_band`:** Q4 and Q5 show sign reversals (SBV negative, SynthDrive
  positive). The v0.1 linear territory score (`territory_beta × (zone − 50)`) cannot
  reproduce the non-monotonic SBV pattern across zones.

- **`car_use` (Commute, Private):** SynthDrive over-suppresses these relativities
  relative to SBV (−0.49 vs −0.23; −0.50 vs −0.24). Categorical variables are sampled
  independently of continuous covariates in v0.1; covariate-mix differences across use
  categories are not corrected in the risk score.

- **`insured_sex` and `marital_status`:** Absolute SBV coefficients are small
  (≤ 0.09 log-RR), but SynthDrive roughly doubles them. Both variables are sampled
  from marginal proportions only; their relationship with claim risk is not explicitly
  calibrated in the risk score function.

---

## 11. Known limitations

- **Synthetic seed data.** The SBV CSV is itself synthetic, generated from a real Canadian
  insurer dataset. SynthDrive cannot recover information lost in that earlier synthesis step.

- **No country-learned behavior.** SynthDrive v0.1 does not encode driving behavior from
  any specific national telematics corpus. The Japan preset (`presets/jp.py`) in v0.2 will
  use Japan-calibrated outcomes (from GIAJ/NLIRO statistics) but will not represent
  Japan-learned driving behavior unless Japanese telematics data is used.

- **Single severity component.** v0.1 uses a single Gamma distribution for severity.
  The heavy tail of real claim distributions (especially large commercial losses) may not
  be fully captured. A mixture model is planned for v0.2.

- **Independent categoricals.** In v0.1, categorical variables (`insured_sex`,
  `marital_status`, `car_use`) are sampled independently of the continuous variables.
  Any cross-variable correlations between, for example, `car_use` and `annual_miles_drive`
  are not explicitly modeled.

- **Mileage in miles.** All mileage columns follow SBV naming conventions and use miles.
  Kilometre variants are planned for the Japan preset in v0.2.

---

## 12. References

So, B., Boucher, J.-P., and Valdez, E. A. (2021). Synthetic Dataset Generation of Driver
Telematics. *Risks*, 9(4), 58. https://doi.org/10.3390/risks9040058
