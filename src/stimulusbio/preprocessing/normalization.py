"""Normalization / transformation of expression data (Step 4).

This is the first stage allowed to change numeric values — ingestion
and validation must not have touched them.

Every transformation applied here must be:
- explicit (named method, explicit parameters)
- recorded (for reporting/provenance, Step 10)
- reversible in documentation, even if not in code
"""

from __future__ import annotations

import pandas as pd


def normalize_expression(
    expression: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Apply a named normalization method to the expression matrix."""
    raise NotImplementedError