"""Tests for the statistics module.

exploratory.compute_pca (Step 6) and differential.compare_conditions
(Step 7) are both implemented and tested below.
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
# differential.compare_conditions
# -------------------------

pytest.importorskip(
    "pydeseq2",
    reason="pydeseq2 (the 'stats' extra) is not installed in this environment.",
)


def _synthetic_counts(
    n_genes=150,
    n_per_condition=5,
    seed=0,
    up_gene="gene_0",
    up_amount=400,
):
    rng = np.random.default_rng(seed)
    samples = [f"sample_{i}" for i in range(2 * n_per_condition)]
    condition = ["control"] * n_per_condition + ["stimulus"] * n_per_condition

    counts = pd.DataFrame(
        rng.poisson(lam=60, size=(n_genes, len(samples))),
        index=[f"gene_{i}" for i in range(n_genes)],
        columns=samples,
    )

    if up_gene is not None:
        stimulus_samples = [s for s, c in zip(samples, condition) if c == "stimulus"]
        counts.loc[up_gene, stimulus_samples] += up_amount

    metadata = pd.DataFrame({"sample_id": samples, "condition": condition})

    return counts, metadata


def test_compare_conditions_returns_expected_columns_and_index():
    counts, metadata = _synthetic_counts()

    result = compare_conditions(counts, metadata, "stimulus", "control")

    assert list(result.columns) == [
        "base_mean",
        "log2_fold_change",
        "lfc_se",
        "p_value",
        "p_value_adj",
    ]
    assert list(result.index) == list(counts.index)


def test_compare_conditions_direction_matches_condition_a_vs_condition_b():
    counts, metadata = _synthetic_counts(up_gene="gene_0", up_amount=400)

    result = compare_conditions(counts, metadata, "stimulus", "control")

    assert result.loc["gene_0", "log2_fold_change"] > 2.0
    assert result.loc["gene_0", "p_value_adj"] < 0.01


def test_compare_conditions_direction_reverses_with_swapped_arguments():
    counts, metadata = _synthetic_counts(up_gene="gene_0", up_amount=400)

    forward = compare_conditions(counts, metadata, "stimulus", "control")
    reversed_ = compare_conditions(counts, metadata, "control", "stimulus")

    assert forward.loc["gene_0", "log2_fold_change"] == pytest.approx(
        -reversed_.loc["gene_0", "log2_fold_change"], rel=1e-6
    )


def test_compare_conditions_reports_uncertainty_alongside_effect_size():
    counts, metadata = _synthetic_counts()

    result = compare_conditions(counts, metadata, "stimulus", "control")

    assert (result["lfc_se"].dropna() > 0).all()


def test_compare_conditions_applies_fdr_correction():
    counts, metadata = _synthetic_counts()

    result = compare_conditions(counts, metadata, "stimulus", "control")

    tested = result.dropna(subset=["p_value", "p_value_adj"])
    assert (tested["p_value_adj"] >= tested["p_value"]).all()


def test_compare_conditions_gives_nan_for_independent_filtering():
    counts, metadata = _synthetic_counts()
    counts.iloc[-1] = 0

    result = compare_conditions(counts, metadata, "stimulus", "control")

    assert pd.isna(result.iloc[-1]["p_value"])
    assert pd.isna(result.iloc[-1]["p_value_adj"])


def test_compare_conditions_does_not_mutate_inputs():
    counts, metadata = _synthetic_counts()
    original_counts = counts.copy(deep=True)
    original_metadata = metadata.copy(deep=True)

    compare_conditions(counts, metadata, "stimulus", "control")

    pd.testing.assert_frame_equal(counts, original_counts)
    pd.testing.assert_frame_equal(metadata, original_metadata)


def test_compare_conditions_rejects_empty_expression():
    counts, metadata = _synthetic_counts()

    with pytest.raises(ValueError, match="empty"):
        compare_conditions(counts.iloc[0:0], metadata, "stimulus", "control")


def test_compare_conditions_rejects_equal_conditions():
    counts, metadata = _synthetic_counts()

    with pytest.raises(ValueError, match="must differ"):
        compare_conditions(counts, metadata, "stimulus", "stimulus")


def test_compare_conditions_rejects_unknown_condition():
    counts, metadata = _synthetic_counts()

    with pytest.raises(ValueError, match="not found"):
        compare_conditions(counts, metadata, "nonexistent", "control")


def test_compare_conditions_rejects_sample_mismatch():
    counts, metadata = _synthetic_counts()
    mismatched_metadata = metadata.iloc[:-1]

    with pytest.raises(ValueError, match="do not match"):
        compare_conditions(counts, mismatched_metadata, "stimulus", "control")


def test_compare_conditions_rejects_missing_values():
    counts, metadata = _synthetic_counts()
    counts.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        compare_conditions(counts, metadata, "stimulus", "control")


def test_compare_conditions_rejects_negative_values():
    counts, metadata = _synthetic_counts()
    counts.iloc[0, 0] = -1

    with pytest.raises(ValueError, match="negative"):
        compare_conditions(counts, metadata, "stimulus", "control")


def test_compare_conditions_rejects_non_whole_number_values():
    counts, metadata = _synthetic_counts()
    counts = counts.astype(float)
    counts.iloc[0, 0] = 1.5

    with pytest.raises(ValueError, match="non-whole-number"):
        compare_conditions(counts, metadata, "stimulus", "control")


def test_compare_conditions_rejects_insufficient_replicates():
    counts, metadata = _synthetic_counts(n_per_condition=1)

    with pytest.raises(ValueError, match="at least 2"):
        compare_conditions(counts, metadata, "stimulus", "control")


def test_compare_conditions_requires_metadata_columns():
    counts, metadata = _synthetic_counts()
    metadata = metadata.drop(columns=["condition"])

    with pytest.raises(ValueError, match="sample_id.*condition"):
        compare_conditions(counts, metadata, "stimulus", "control")


def test_compare_conditions_gives_actionable_error_without_pydeseq2(monkeypatch):
    """Simulates the optional 'stats' extra being absent."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pydeseq2"):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    counts, metadata = _synthetic_counts()

    with pytest.raises(ImportError, match="stats"):
        compare_conditions(counts, metadata, "stimulus", "control")
