"""Tests for the statistics module.

exploratory.compute_pca (Step 6) is implemented and tested below.

differential.compare_conditions (Step 7) is still a stub. It is
deliberately unresolved pending an open decision (project doc, section
28): whether it wraps an established DE method (e.g. rpy2 ->
DESeq2/edgeR) or reimplements DE statistics from scratch. That needs
its own dedicated chat — see differential.py's module docstring.
"""

import numpy as np
import pandas as pd
import pytest

from stimulusbio.statistics.differential import compare_conditions
from stimulusbio.statistics.exploratory import compute_pca

# -------------------------
# compute_pca: shape / basic contract
# -------------------------

def test_pca_returns_dataframe():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0, 4.0],
            "sample_2": [2.0, 3.0, 4.0, 5.0],
            "sample_3": [10.0, 1.0, 6.0, 2.0],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )

    result = compute_pca(expression)

    assert isinstance(result, pd.DataFrame)


def test_pca_default_n_components_is_two():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0, 4.0],
            "sample_2": [2.0, 3.0, 4.0, 5.0],
            "sample_3": [10.0, 1.0, 6.0, 2.0],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )

    result = compute_pca(expression)

    assert list(result.columns) == ["PC1", "PC2"]


def test_pca_custom_n_components():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0, 4.0],
            "sample_2": [2.0, 3.0, 4.0, 5.0],
            "sample_3": [10.0, 1.0, 6.0, 2.0],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )

    result = compute_pca(expression, n_components=3)

    assert list(result.columns) == ["PC1", "PC2", "PC3"]


def test_pca_row_count_matches_sample_count():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0],
            "sample_2": [2.0, 3.0, 4.0],
            "sample_3": [10.0, 1.0, 6.0],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    result = compute_pca(expression, n_components=1)

    assert len(result) == 3


def test_pca_index_is_sample_ids():
    expression = pd.DataFrame(
        {
            "Sample_A": [1.0, 2.0, 3.0, 4.0],
            "Sample_B": [2.0, 3.0, 4.0, 5.0],
            "Sample_C": [10.0, 1.0, 6.0, 2.0],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )

    result = compute_pca(expression)

    assert list(result.index) == ["Sample_A", "Sample_B", "Sample_C"]
    assert result.index.name == "sample_id"


def test_pca_does_not_mutate_input():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0, 4.0],
            "sample_2": [2.0, 3.0, 4.0, 5.0],
            "sample_3": [10.0, 1.0, 6.0, 2.0],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )
    original = expression.copy(deep=True)

    compute_pca(expression)

    pd.testing.assert_frame_equal(expression, original)


# -------------------------
# compute_pca: explained variance metadata
# -------------------------

def test_pca_explained_variance_ratio_attached():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0, 4.0],
            "sample_2": [2.0, 3.0, 4.0, 5.0],
            "sample_3": [10.0, 1.0, 6.0, 2.0],
        },
        index=["gene_1", "gene_2", "gene_3", "gene_4"],
    )

    result = compute_pca(expression, n_components=2)

    ratios = result.attrs["explained_variance_ratio"]

    assert len(ratios) == 2
    assert all(0.0 <= ratio <= 1.0 for ratio in ratios)
    assert ratios[0] >= ratios[1]


# -------------------------
# compute_pca: numeric correctness on a known case
# -------------------------

def test_pca_first_component_captures_dominant_variance():
    # Two features perfectly correlated (feature_2 = 2 * feature_1) carry
    # all the variance; a third, constant feature carries none. PC1 should
    # explain (almost) all of it.
    expression = pd.DataFrame(
        {
            "sample_1": [0.0, 0.0, 5.0],
            "sample_2": [1.0, 2.0, 5.0],
            "sample_3": [2.0, 4.0, 5.0],
            "sample_4": [3.0, 6.0, 5.0],
        },
        index=["feature_1", "feature_2", "feature_3"],
    )

    result = compute_pca(expression, n_components=2)

    ratios = result.attrs["explained_variance_ratio"]

    assert ratios[0] == pytest.approx(1.0, abs=1e-6)


# -------------------------
# compute_pca: validation / error handling
# -------------------------

def test_pca_empty_matrix_is_rejected():
    expression = pd.DataFrame()

    with pytest.raises(ValueError, match="expression matrix is empty"):
        compute_pca(expression)


def test_pca_missing_values_are_rejected():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, np.nan],
            "sample_2": [2.0, 3.0],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="missing values"):
        compute_pca(expression)


def test_pca_non_numeric_values_are_rejected():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, "invalid"],
            "sample_2": [2.0, 3.0],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="non-numeric values"):
        compute_pca(expression)


def test_pca_zero_n_components_is_rejected():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0],
            "sample_2": [2.0, 3.0],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="n_components must be at least 1"):
        compute_pca(expression, n_components=0)


def test_pca_negative_n_components_is_rejected():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0],
            "sample_2": [2.0, 3.0],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="n_components must be at least 1"):
        compute_pca(expression, n_components=-1)


def test_pca_too_many_components_is_rejected():
    # 2 samples, 3 features -> max supported components is 2.
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0],
            "sample_2": [4.0, 5.0, 6.0],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    with pytest.raises(ValueError, match="exceeds the maximum supported"):
        compute_pca(expression, n_components=3)


def test_pca_n_components_can_equal_max_supported():
    expression = pd.DataFrame(
        {
            "sample_1": [1.0, 2.0, 3.0],
            "sample_2": [4.0, 5.0, 6.0],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    result = compute_pca(expression, n_components=2)

    assert list(result.columns) == ["PC1", "PC2"]


# -------------------------
# differential.compare_conditions: intentionally still a stub
# -------------------------

@pytest.mark.skip(
    reason=(
        "Step 7 blocked on an open design decision (project doc, section "
        "28): rpy2-wrapped DESeq2/edgeR vs. a from-scratch implementation. "
        "Needs its own dedicated chat."
    )
)
def test_compare_conditions_placeholder():
    pass


def test_compare_conditions_is_still_unimplemented():
    """Guard test: fails loudly once someone implements this, as a nudge
    to also un-skip/replace the placeholder above and update docs."""
    expression = pd.DataFrame(
        {"sample_1": [1.0], "sample_2": [2.0]}, index=["gene_1"]
    )
    metadata = pd.DataFrame(
        {"sample_id": ["sample_1", "sample_2"], "condition": ["control", "stimulus"]}
    )

    with pytest.raises(NotImplementedError):
        compare_conditions(expression, metadata, "control", "stimulus")
