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
    """Remove features below a minimum mean expression threshold."""
    raise NotImplementedError