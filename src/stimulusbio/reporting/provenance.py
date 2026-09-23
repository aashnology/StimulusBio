"""Provenance tracking (Step 10): parameters, environment, versions.

Captures what the pipeline needs to be reproducible — not the results
themselves. This module should have no dependency on the analysis
modules; it only records what was called and with what arguments.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

# Matches the runtime dependencies declared in pyproject.toml.
# scikit-learn's distribution name differs from its import name.
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "matplotlib",
    "seaborn",
)


def capture_environment() -> dict[str, str]:
    """Return package versions and Python version used for a run.

    Looks up each of this project's declared runtime dependencies via
    installed-package metadata (not by importing the packages), plus
    the running Python version. A dependency that is declared but not
    actually installed is recorded as ``"not installed"`` rather than
    raising, so this function never fails just because an optional
    package is absent.
    """
    environment: dict[str, str] = {"python": sys.version.split()[0]}

    for package_name in TRACKED_PACKAGES:
        try:
            environment[package_name] = version(package_name)
        except PackageNotFoundError:
            environment[package_name] = "not installed"

    return environment
