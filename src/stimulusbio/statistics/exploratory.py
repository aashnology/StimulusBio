"""Exploratory analysis (Step 6): PCA, clustering, correlation.

Purely descriptive — no hypothesis testing, no condition comparisons
here. That belongs in differential.py.
"""

from __future__ import annotations

import pandas as pd


def compute_pca(expression: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Return principal component scores per sample."""
    raise NotImplementedError