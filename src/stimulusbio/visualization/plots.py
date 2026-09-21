"""Reproducible figure generation (Step 9).

Every function here takes data in, returns a matplotlib Figure out.
No function should read from disk or depend on global state — this
keeps figures independently testable and embeddable in reports.
"""

from __future__ import annotations

import matplotlib.figure


def plot_pca(pca_scores, metadata) -> matplotlib.figure.Figure:
    """Scatter plot of PCA scores, colored by condition."""
    raise NotImplementedError