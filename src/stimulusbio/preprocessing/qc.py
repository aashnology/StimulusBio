"""Quality-control diagnostics (Step 5).

QC functions only compute and report — they must not modify or drop
data themselves. Any resulting exclusions are a decision made outside
this module, driven by the caller.
"""

from __future__ import annotations

import pandas as pd

SUMMARY_COLUMNS = [
    "library_size",
    "n_missing",
    "pct_missing",
    "n_zero",
    "pct_zero",
]


def compute_sample_summary(expression: pd.DataFrame) -> pd.DataFrame:
    """Per-sample QC metrics: library size, missingness, etc.

    Returns one row per sample (column of ``expression``) with:

    - ``library_size``: sum of expression values for that sample
      (missing values excluded from the sum).
    - ``n_missing`` / ``pct_missing``: count and percentage of features
      with a missing value in that sample.
    - ``n_zero`` / ``pct_zero``: count and percentage of features with
      an exact-zero value in that sample.

    This function only computes and reports. It never modifies,
    filters, or reorders ``expression`` — any resulting exclusions are
    a decision made by the caller, outside this module.
    """
    if expression.empty:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        summary.index.name = "sample_id"
        return summary

    n_features = len(expression)

    library_size = expression.sum(axis=0, skipna=True)
    n_missing = expression.isna().sum(axis=0)
    n_zero = (expression == 0).sum(axis=0)

    summary = pd.DataFrame(
        {
            "library_size": library_size,
            "n_missing": n_missing,
            "pct_missing": (n_missing / n_features) * 100,
            "n_zero": n_zero,
            "pct_zero": (n_zero / n_features) * 100,
        }
    )
    summary.index.name = "sample_id"

    return summary