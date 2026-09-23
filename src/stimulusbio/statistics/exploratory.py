"""Exploratory analysis (Step 6): PCA, clustering, correlation.

Purely descriptive — no hypothesis testing, no condition comparisons
here. That belongs in differential.py.
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA


def compute_pca(expression: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Return principal component scores per sample.

    ``expression`` is a feature-by-sample matrix, matching the layout
    produced by ``io.expression.load_expression_matrix``: rows are
    features, columns are samples. PCA treats samples as observations
    and features as variables, so the matrix is transposed internally
    before fitting.

    Only mean-centering is applied (each feature centered across
    samples) — no scaling to unit variance. Whether to scale features
    first (e.g. z-scoring via
    ``preprocessing.normalization.normalize_expression``) is a
    modelling choice left to the caller; this keeps PCA's own contract
    simple and its input explicit rather than silently rescaling data.

    Raises ``ValueError`` if:

    - the expression matrix is empty
    - the expression matrix contains non-numeric values
    - the expression matrix contains missing values
    - ``n_components`` is less than 1
    - ``n_components`` exceeds ``min(n_samples, n_features)``

    The returned DataFrame has one row per sample (indexed by
    ``sample_id``, taken from ``expression.columns``) and columns
    ``PC1..PCn``. Per-component explained-variance ratios are attached
    at ``result.attrs["explained_variance_ratio"]`` for convenience —
    they are supplementary metadata, not part of the tabular values.

    This function only computes and reports. It never modifies
    ``expression``.
    """
    if expression.empty:
        raise ValueError("Cannot compute PCA: expression matrix is empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression.dtypes):
        raise ValueError(
            "Cannot compute PCA: expression matrix contains non-numeric values."
        )

    if expression.isna().any().any():
        raise ValueError(
            "Cannot compute PCA: expression matrix contains missing values."
        )

    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}.")

    samples_by_features = expression.T
    n_samples, n_features = samples_by_features.shape
    max_components = min(n_samples, n_features)

    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds the maximum supported by "
            f"this data: min(n_samples={n_samples}, n_features={n_features}) "
            f"= {max_components}."
        )

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(samples_by_features.to_numpy())

    result = pd.DataFrame(
        scores,
        # Copy the index: pandas' transpose can share the underlying
        # Index object with the original (homogeneous-dtype) frame, and
        # naming it below must not mutate the caller's ``expression``.
        index=samples_by_features.index.copy(),
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )
    result.index.name = "sample_id"
    result.attrs["explained_variance_ratio"] = tuple(pca.explained_variance_ratio_)

    return result
