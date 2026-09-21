"""Human-readable + machine-readable report assembly (Step 10).

Combines validation results, QC summaries, statistical results, and
provenance into a single output. Does not compute anything itself —
pure assembly.
"""

from __future__ import annotations


def build_report(results: dict, provenance: dict) -> dict:
    """Assemble a structured report dict from pipeline outputs."""
    raise NotImplementedError