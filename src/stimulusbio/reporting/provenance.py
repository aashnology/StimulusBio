"""Provenance tracking (Step 10): parameters, environment, versions.

Captures what the pipeline needs to be reproducible — not the results
themselves. This module should have no dependency on the analysis
modules; it only records what was called and with what arguments.
"""

from __future__ import annotations


def capture_environment() -> dict[str, str]:
    """Return package versions and Python version used for a run."""
    raise NotImplementedError