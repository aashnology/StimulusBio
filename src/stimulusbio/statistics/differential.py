"""Differential analysis between experimental conditions (Step 7).

Must report effect size + uncertainty alongside significance, and
apply multiple-testing correction (FDR) by default.

Open decision (per project doc, section 28): whether this wraps an
established method (e.g. via rpy2 to DESeq2/edgeR) rather than
reimplementing differential-expression statistics from scratch.
Do not resolve this silently — it needs its own dedicated chat.
"""

from __future__ import annotations

import pandas as pd


def compare_conditions(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    """Return per-feature effect size, p-value, and adjusted p-value."""
    raise NotImplementedError