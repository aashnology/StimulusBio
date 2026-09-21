"""Quality-control diagnostics (Step 5).

QC functions only compute and report — they must not modify or drop
data themselves. Any resulting exclusions are a decision made outside
this module, driven by the caller.
"""

from __future__ import annotations

import pandas as pd


def compute_sample_summary(expression: pd.DataFrame) -> pd.DataFrame:
    """Per-sample QC metrics: library size, missingness, etc."""
    raise NotImplementedError