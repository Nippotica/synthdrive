"""
Pytest fixtures for SynthDrive v0.1 tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthdrive
from synthdrive.presets.core import load_core_preset


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def core_preset():
    return load_core_preset()


@pytest.fixture(scope="session")
def small_df():
    """Generate a small dataset once per test session."""
    return synthdrive.generate(n=500, random_state=99)


@pytest.fixture(scope="session")
def medium_df():
    """Generate a medium dataset once per test session."""
    return synthdrive.generate(n=5_000, random_state=42)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
