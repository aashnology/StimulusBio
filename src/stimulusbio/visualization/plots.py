"""Reproducible figure generation (Step 9).

Every function here takes data in, returns a matplotlib Figure out.
No function should read from disk or depend on global state — this
keeps figures independently testable and embeddable in reports.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure


def plot_pca(pca_scores: pd.DataFrame, metadata: pd.DataFrame) -> Figure:
    """Scatter plot of PCA scores, colored by condition.

    ``pca_scores`` is the output of
    ``statistics.exploratory.compute_pca``: indexed by ``sample_id``,
    with columns ``PC1..PCn``. Only ``PC1`` and ``PC2`` are plotted.
    If ``pca_scores.attrs["explained_variance_ratio"]`` is present,
    the first two values are used to label the axes.

    ``metadata`` must contain ``sample_id`` and ``condition`` columns
    (matching ``io.metadata.load_metadata``'s contract); it is used
    only to color points by condition.

    Builds the figure directly via ``matplotlib.figure.Figure`` rather
    than ``pyplot``, so no global matplotlib state is touched and no
    figure is left open. Neither input is read from disk or mutated.

    Raises ``ValueError`` if ``pca_scores`` is missing ``PC1``/``PC2``,
    if ``metadata`` is missing ``sample_id``/``condition``, or if any
    sample in ``pca_scores`` has no matching row in ``metadata``.
    """
    required_pcs = {"PC1", "PC2"}
    missing_pcs = required_pcs - set(pca_scores.columns)

    if missing_pcs:
        raise ValueError(
            "plot_pca requires 'PC1' and 'PC2' columns in pca_scores; "
            f"missing: {sorted(missing_pcs)}."
        )

    required_metadata_columns = {"sample_id", "condition"}
    missing_columns = required_metadata_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "metadata must contain 'sample_id' and 'condition' columns; "
            f"missing: {sorted(missing_columns)}."
        )

    condition_by_sample = metadata.set_index("sample_id")["condition"]
    missing_samples = [
        sample_id
        for sample_id in pca_scores.index
        if sample_id not in condition_by_sample.index
    ]

    if missing_samples:
        raise ValueError(
            "Samples in pca_scores missing from metadata: "
            f"{sorted(missing_samples)}."
        )

    conditions = condition_by_sample.loc[pca_scores.index]

    figure = Figure(figsize=(6.0, 5.0))
    axis = figure.add_subplot(111)

    for condition, group in pca_scores.groupby(conditions):
        axis.scatter(group["PC1"], group["PC2"], label=str(condition), alpha=0.8)

    explained_variance_ratio = pca_scores.attrs.get("explained_variance_ratio")

    if explained_variance_ratio and len(explained_variance_ratio) >= 2:
        axis.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)")
    else:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")

    axis.set_title("PCA of samples by condition")
    axis.legend(title="condition")

    return figure
