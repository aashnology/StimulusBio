import pandas as pd
import pytest

from stimulusbio.preprocessing.filtering import filter_low_expression_features
from stimulusbio.preprocessing.normalization import normalize_expression
from stimulusbio.preprocessing.qc import compute_sample_summary

# -------------------------
# QC: compute_sample_summary
# -------------------------

def test_qc_computes_library_size():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20, 30],
            "sample_2": [1, 2, 3],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    summary = compute_sample_summary(expression)

    assert summary.loc["sample_1", "library_size"] == 60
    assert summary.loc["sample_2", "library_size"] == 6


def test_qc_detects_missing_values_per_sample():
    expression = pd.DataFrame(
        {
            "sample_1": [10, None, 30],
            "sample_2": [1, 2, 3],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    summary = compute_sample_summary(expression)

    assert summary.loc["sample_1", "n_missing"] == 1
    assert summary.loc["sample_2", "n_missing"] == 0
    assert summary.loc["sample_1", "pct_missing"] == pytest.approx(100 / 3)


def test_qc_detects_zero_values_per_sample():
    expression = pd.DataFrame(
        {
            "sample_1": [0, 0, 30],
            "sample_2": [1, 2, 3],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    summary = compute_sample_summary(expression)

    assert summary.loc["sample_1", "n_zero"] == 2
    assert summary.loc["sample_2", "n_zero"] == 0
    assert summary.loc["sample_1", "pct_zero"] == pytest.approx(200 / 3)


def test_qc_returns_one_row_per_sample_in_column_order():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20],
            "sample_2": [1, 2],
            "sample_3": [5, 6],
        },
        index=["gene_1", "gene_2"],
    )

    summary = compute_sample_summary(expression)

    assert list(summary.index) == ["sample_1", "sample_2", "sample_3"]


def test_qc_does_not_mutate_input():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20],
            "sample_2": [1, 2],
        },
        index=["gene_1", "gene_2"],
    )
    original = expression.copy(deep=True)

    compute_sample_summary(expression)

    pd.testing.assert_frame_equal(expression, original)


def test_qc_handles_empty_expression_matrix():
    expression = pd.DataFrame()

    summary = compute_sample_summary(expression)

    assert summary.empty


# -------------------------
# Filtering: filter_low_expression_features
# -------------------------

def test_filter_removes_features_below_threshold():
    expression = pd.DataFrame(
        {
            "sample_1": [1, 10, 100],
            "sample_2": [1, 10, 100],
        },
        index=["low_gene", "mid_gene", "high_gene"],
    )

    filtered, removed = filter_low_expression_features(expression, min_mean=5.0)

    assert "low_gene" not in filtered.index
    assert removed == ["low_gene"]


def test_filter_keeps_features_at_or_above_threshold():
    expression = pd.DataFrame(
        {
            "sample_1": [5, 10],
            "sample_2": [5, 10],
        },
        index=["gene_1", "gene_2"],
    )

    filtered, removed = filter_low_expression_features(expression, min_mean=5.0)

    assert list(filtered.index) == ["gene_1", "gene_2"]
    assert removed == []


def test_filter_never_drops_samples():
    expression = pd.DataFrame(
        {
            "sample_1": [1, 100],
            "sample_2": [1, 100],
            "sample_3": [1, 100],
        },
        index=["low_gene", "high_gene"],
    )

    filtered, _ = filter_low_expression_features(expression, min_mean=5.0)

    assert list(filtered.columns) == ["sample_1", "sample_2", "sample_3"]


def test_filter_preserves_values_for_retained_features():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20],
            "sample_2": [30, 40],
        },
        index=["gene_1", "gene_2"],
    )

    filtered, _ = filter_low_expression_features(expression, min_mean=0.0)

    pd.testing.assert_frame_equal(filtered, expression)


def test_filter_does_not_mutate_input():
    expression = pd.DataFrame(
        {
            "sample_1": [1, 100],
            "sample_2": [1, 100],
        },
        index=["low_gene", "high_gene"],
    )
    original = expression.copy(deep=True)

    filter_low_expression_features(expression, min_mean=5.0)

    pd.testing.assert_frame_equal(expression, original)


def test_filter_removed_list_preserves_original_order():
    expression = pd.DataFrame(
        {
            "sample_1": [1, 100, 2],
        },
        index=["gene_a", "gene_b", "gene_c"],
    )

    _, removed = filter_low_expression_features(expression, min_mean=5.0)

    assert removed == ["gene_a", "gene_c"]


def test_filter_handles_empty_expression_matrix():
    expression = pd.DataFrame()

    filtered, removed = filter_low_expression_features(expression, min_mean=5.0)

    assert filtered.empty
    assert removed == []


# -------------------------
# Normalization: normalize_expression
# -------------------------

def test_normalize_cpm_scales_to_one_million_per_sample():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 90],
            "sample_2": [50, 50],
        },
        index=["gene_1", "gene_2"],
    )

    normalized = normalize_expression(expression, method="cpm")

    assert normalized["sample_1"].sum() == pytest.approx(1_000_000)
    assert normalized["sample_2"].sum() == pytest.approx(1_000_000)
    assert normalized.loc["gene_1", "sample_1"] == pytest.approx(100_000)


def test_normalize_cpm_rejects_zero_library_size():
    expression = pd.DataFrame(
        {
            "sample_1": [0, 0],
            "sample_2": [10, 20],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="zero library size"):
        normalize_expression(expression, method="cpm")


def test_normalize_log2_applies_log_plus_one_transform():
    expression = pd.DataFrame(
        {
            "sample_1": [0, 3],
        },
        index=["gene_1", "gene_2"],
    )

    normalized = normalize_expression(expression, method="log2")

    assert normalized.loc["gene_1", "sample_1"] == pytest.approx(0.0)
    assert normalized.loc["gene_2", "sample_1"] == pytest.approx(2.0)


def test_normalize_log2_rejects_negative_values():
    expression = pd.DataFrame(
        {
            "sample_1": [-1, 3],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="negative values"):
        normalize_expression(expression, method="log2")


def test_normalize_zscore_centers_and_scales_each_feature():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 15],
            "sample_2": [20, 20],
            "sample_3": [30, 25],
        },
        index=["gene_1", "gene_2"],
    )

    normalized = normalize_expression(expression, method="zscore")

    assert normalized.loc["gene_1"].mean() == pytest.approx(0.0, abs=1e-10)
    assert normalized.loc["gene_1"].std() == pytest.approx(1.0)


def test_normalize_zscore_rejects_zero_variance_feature():
    expression = pd.DataFrame(
        {
            "sample_1": [5, 5],
            "sample_2": [5, 10],
        },
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="zero-variance"):
        normalize_expression(expression, method="zscore")


def test_normalize_rejects_unsupported_method():
    expression = pd.DataFrame(
        {"sample_1": [1, 2]},
        index=["gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="Unsupported normalization method"):
        normalize_expression(expression, method="quantile")


def test_normalize_does_not_mutate_input():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 90],
            "sample_2": [50, 50],
        },
        index=["gene_1", "gene_2"],
    )
    original = expression.copy(deep=True)

    normalize_expression(expression, method="cpm")

    pd.testing.assert_frame_equal(expression, original)


def test_normalize_preserves_shape_and_labels():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 90],
            "sample_2": [50, 50],
        },
        index=["gene_1", "gene_2"],
    )

    normalized = normalize_expression(expression, method="cpm")

    assert list(normalized.index) == ["gene_1", "gene_2"]
    assert list(normalized.columns) == ["sample_1", "sample_2"]


def test_normalize_handles_empty_expression_matrix():
    expression = pd.DataFrame()

    normalized = normalize_expression(expression, method="cpm")

    assert normalized.empty
