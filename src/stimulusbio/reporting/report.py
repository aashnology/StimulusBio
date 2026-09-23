"""Human-readable + machine-readable report assembly (Step 10).

Combines validation results, QC summaries, statistical results, and
provenance into a single output. Does not compute anything itself —
pure assembly.
"""

from __future__ import annotations

from typing import Any


def build_report(results: dict[str, Any], provenance: dict[str, Any]) -> dict[str, object]:
    """Assemble a structured report dict from pipeline outputs.

    Pure assembly only — this function computes nothing. ``results``
    is a dict of already-computed, named pipeline outputs (e.g.
    ``{"validation": validate_dataset(...), "qc": compute_sample_summary(...),
    "differential_expression": compare_conditions(...)}``); each entry
    is included in the report exactly as given, with no interpretation,
    validation, or reformatting of its contents. ``provenance`` is
    attached under its own top-level key, typically the output of
    ``reporting.provenance.capture_environment`` plus whatever run
    parameters the caller wants recorded.

    Neither input is mutated: the returned dict holds shallow copies of
    both, so adding or removing top-level keys on the report afterward
    does not affect the caller's original dicts (mutating a value
    shared by reference, e.g. a DataFrame inside ``results``, still
    affects both — this function only guards its own top-level
    structure, not deep copies of arbitrary result objects).

    Raises ``TypeError`` if ``results`` or ``provenance`` is not a
    dict.
    """
    if not isinstance(results, dict):
        raise TypeError(f"results must be a dict, got {type(results).__name__}.")

    if not isinstance(provenance, dict):
        raise TypeError(f"provenance must be a dict, got {type(provenance).__name__}.")

    return {
        "results": dict(results),
        "provenance": dict(provenance),
    }
