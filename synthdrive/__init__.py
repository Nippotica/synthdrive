"""
SynthDrive v0.1 — Actuarially structured synthetic telematics dataset generator.

Quick start
-----------
    import synthdrive

    # Parameter mode (no CSV required)
    df = synthdrive.generate(n=100_000, random_state=42)

    # Seed mode (So–Boucher–Valdez CSV)
    df = synthdrive.generate(n=100_000, seed_path="/data/sbv.csv", random_state=42)

    # Validate
    report = synthdrive.validate(df)
    print(report.summary())

Reference
---------
So, Boucher, and Valdez (2021). Synthetic Dataset Generation of Driver
Telematics. Risks 9(4): 58. https://doi.org/10.3390/risks9030058
"""

from synthdrive.generate import generate
from synthdrive.validate.report import validate, ValidationReport

__version__ = "0.1.0"
__all__ = ["generate", "validate", "ValidationReport", "__version__"]
