# SynthDrive v0.1

**Actuarially structured synthetic telematics dataset generator.**

SynthDrive generates synthetic personal auto telematics portfolios with
plausible joint distributions, risk relativities, and constraint-compliant
output. It is calibrated to match the structure and statistics of the
So–Boucher–Valdez (2021) dataset and runs without requiring access to
that dataset.

---

## Quick start

```python
import synthdrive

# Parameter mode — no data file needed
df = synthdrive.generate(n=100_000, random_state=42)

# Seed mode — fit from local SBV CSV
df = synthdrive.generate(n=100_000, seed_path="/data/sbv.csv", random_state=42)

# Validate
report = synthdrive.validate(df)
print(report.summary())
report.save("output/")
```

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and:
`numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`

## Operating modes

| Mode | Requirement | How to use |
|---|---|---|
| Parameter | Nothing | `generate(n=N)` |
| Seed | Local CSV file | `generate(n=N, seed_path="/path/to/sbv.csv")` |

Parameter mode is the default. If `seed_path` is provided but the file is
missing or unreadable, SynthDrive **falls back to parameter mode with a warning**
rather than raising an error.

## Output schema

55 columns covering:
- **Policy**: `policy_id`, `duration`, `exposure`
- **Driver**: `insured_age`, `insured_sex`, `marital_status`, `credit_score`, `years_no_claims`
- **Vehicle**: `car_age`, `car_use`, `territory`, `region`, `annual_miles_drive`
- **Telematics**: mileage, day-of-week, time-of-day, hard-event, turn-intensity variables
- **Claims**: `claim_count`, `claim_amount`, `pure_premium`

All `pct_*` variables are proportions on **[0.0, 1.0]**.

## Tests

```bash
pytest tests/ -v
```

## Reference

So, Boucher, and Valdez (2021). Synthetic Dataset Generation of Driver
Telematics. *Risks* 9(4): 58. https://doi.org/10.3390/risks9040058

## Documentation

- `docs/synthdrive_formal_v0.1.pdf` — Formal algebraic specification
- `docs/synthdrive_techpaper.pdf` — SSRN companion paper
- `docs/methodology_v01.md` — Model specification and parameter calibration
- `docs/limitations.md` — Known limitations