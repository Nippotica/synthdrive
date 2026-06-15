"""
Japan preset stub for SynthDrive v0.2+.

This module is a placeholder.  The Japan preset will calibrate marginal
distributions and the frequency/severity model to Japanese personal auto
insurance statistics, including:
    - Right-hand traffic (affects turn-intensity sign conventions)
    - Lower annual mileage (typical urban Japanese driver: 5 000–8 000 km)
    - Different vehicle age distribution (regular shaken inspection cycles)
    - Lower base claim frequency (~3–4% per year, historically)
    - Smaller mean severity (different tort/compensation system)

The Japan preset requires dedicated calibration data and will be released
in SynthDrive v0.2.
"""

from __future__ import annotations


def load_jp_preset() -> None:
    raise NotImplementedError(
        "The Japan preset is not yet implemented in SynthDrive v0.1. "
        "It is planned for v0.2.  Use preset='core' instead."
    )
