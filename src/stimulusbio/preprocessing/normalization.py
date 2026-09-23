"""Normalization / transformation of expression data (Step 4).

This is the first stage allowed to change numeric values — ingestion
and validation must not have touched them.

Every transformation applied here must be:
- explicit (named method, explicit parameters)
- recorded (for reporting/provenance, Step 10)
- reversible in documentation, even if not in code

Scope note: this module implements general-purpose, method-agnostic
transforms only (CPM scaling, log2 transform, per-feature z-score),
intended for exploratory analysis, QC, and visualization.

RESOLVED (see statistics/differential.py): differential expression
wraps PyDESeq2, which computes its own DESeq2 median-of-ratios size
factors internally from raw counts. Consequently, none of the
transforms in this module (cpm, log2, zscore) are valid inputs to
statistics.differential.compare_conditions — feeding it
already-normalized values would double-normalize and invalidate the
result. compare_conditions takes raw (un-normalized, non-negative
integer) counts directly; this module's outputs are for
exploratory/QC/visualization use only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_METHODS = {"cpm", "log2", "zscore"}


def normalize_expression(
    expression: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Apply a named normalization method to the expression matrix.

    Supported methods:

    - ``"cpm"``: counts-per-million. Each sample (column) is divided
      by its own library size (column sum) and scaled by 1e6. Fails
      loudly if any sample has a library size of zero.
    - ``"log2"``: log2(x + 1) transform, applied element-wise. Fails
      loudly if the matrix contains negative values.
    - ``"zscore"``: per-feature (row-wise) standardization to mean 0,
      standard deviation 1. Fails loudly on any zero-variance feature,
      since it cannot be standardized.

    ``expression`` is never mutated in place. Shape, feature IDs, and
    sample IDs are preserved.
    """
    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(
            f"Unsupported normalization method: '{method}'. "
            f"Supported methods are: {supported}."
        )

    if expression.empty:
        return expression.copy()

    if method == "cpm":
        library_sizes = expression.sum(axis=0, skipna=True)
        zero_library = library_sizes[library_sizes == 0].index.tolist()

        if zero_library:
            raise ValueError(
                "Cannot compute CPM: sample(s) with zero library size: "
                f"{zero_library}."
            )

        normalized = expression.div(library_sizes, axis=1) * 1e6

    elif method == "log2":
        if (expression < 0).any().any():
            raise ValueError(
                "Cannot apply log2 transform: expression matrix contains "
                "negative values."
            )

        normalized = np.log2(expression + 1)

    else:  # method == "zscore"
        row_mean = expression.mean(axis=1, skipna=True)
        row_std = expression.std(axis=1, skipna=True)
        zero_variance = row_std[row_std == 0].index.tolist()

        if zero_variance:
            raise ValueError(
                "Cannot compute z-score: zero-variance feature(s): "
                f"{zero_variance}."
            )

        normalized = expression.sub(row_mean, axis=0).div(row_std, axis=0)

    return normalized