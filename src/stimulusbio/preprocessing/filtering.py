"""Feature-level filtering of expression data (Step 4).

Responsibilities:
- remove low-count / low-variance features according to an explicit,
  documented threshold
- never silently drop samples
- return both the filtered matrix and a record of what was removed

Must NOT:
- normalize or transform values
- infer thresholds from the data without the caller specifying a rule
"""

from __future__ import annotations

import pandas as pd


def filter_low_expression_features(
    expression: pd.DataFrame,
    min_mean: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features below a minimum mean expression threshold.

    A feature is kept when its mean expression across all samples is
    greater than or equal to ``min_mean``. ``min_mean`` is an explicit,
    caller-supplied threshold — no threshold is ever inferred from the
    data here.

    Returns a tuple of (filtered expression matrix, list of removed
    feature IDs, in their original matrix order). ``expression`` is
    never mutated in place, and no sample (column) is ever dropped.
    """
    if expression.empty:
        return expression.copy(), []

    feature_means = expression.mean(axis=1, skipna=True)
    keep_mask = feature_means >= min_mean

    removed = expression.index[~keep_mask].tolist()
    filtered = expression.loc[keep_mask].copy()

    return filtered, removed