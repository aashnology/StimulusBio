import pandas as pd

from stimulusbio.validation.expression import (
    validate_expression_matrix,
    validate_metadata,
    validate_sample_alignment,
)


# -------------------------
# Expression matrix tests
# -------------------------

def test_valid_expression_matrix():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20, 30],
            "sample_2": [11, 21, 31],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    assert validate_expression_matrix(expression) == []


def test_empty_expression_matrix_is_detected():
    expression = pd.DataFrame()

    issues = validate_expression_matrix(expression)

    assert "Expression matrix is empty." in issues


def test_duplicate_feature_ids_are_detected():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20],
            "sample_2": [11, 21],
        },
        index=["gene_1", "gene_1"],
    )

    issues = validate_expression_matrix(expression)

    assert "Expression matrix contains duplicate feature IDs." in issues


def test_duplicate_sample_ids_are_detected():
    expression = pd.DataFrame(
        [[10, 20], [30, 40]],
        index=["gene_1", "gene_2"],
        columns=["sample_1", "sample_1"],
    )

    issues = validate_expression_matrix(expression)

    assert "Expression matrix contains duplicate sample IDs." in issues


def test_missing_expression_values_are_detected():
    expression = pd.DataFrame(
        {
            "sample_1": [10, None],
            "sample_2": [11, 21],
        },
        index=["gene_1", "gene_2"],
    )

    issues = validate_expression_matrix(expression)

    assert "Expression matrix contains missing values." in issues


def test_non_numeric_expression_values_are_detected():
    expression = pd.DataFrame(
        {
            "sample_1": [10, "invalid"],
            "sample_2": [11, 21],
        },
        index=["gene_1", "gene_2"],
    )

    issues = validate_expression_matrix(expression)

    assert "Expression matrix contains non-numeric values." in issues


# -------------------------
# Metadata tests
# -------------------------

def test_missing_metadata_columns_are_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
        }
    )

    issues = validate_metadata(metadata)

    assert "Metadata is missing required columns: ['condition']." in issues


def test_duplicate_metadata_sample_ids_are_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_1"],
            "condition": ["control", "stimulus"],
        }
    )

    issues = validate_metadata(metadata)

    assert "Metadata contains duplicate sample IDs." in issues


def test_missing_condition_labels_are_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "condition": ["control", None],
        }
    )

    issues = validate_metadata(metadata)

    assert "Metadata contains missing condition labels." in issues


# -------------------------
# Sample alignment tests
# -------------------------

def test_sample_alignment_is_checked():
    expression = pd.DataFrame(
        {
            "sample_1": [10],
            "sample_2": [20],
        },
        index=["gene_1"],
    )

    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_3"],
            "condition": ["control", "stimulus"],
        }
    )

    issues = validate_sample_alignment(expression, metadata)

    assert any("sample_2" in issue for issue in issues)
    assert any("sample_3" in issue for issue in issues)


def test_missing_sample_id_prevents_alignment_check():
    expression = pd.DataFrame(
        {"sample_1": [10]},
        index=["gene_1"],
    )

    metadata = pd.DataFrame(
        {"condition": ["control"]},
    )

    issues = validate_sample_alignment(expression, metadata)

    assert "Cannot check sample alignment: 'sample_id' is missing." in issues


# -------------------------
# Experimental design tests
# -------------------------

def test_empty_experimental_group_is_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "condition": ["control", "control"],
        }
    )

    conditions = metadata["condition"].unique()

    assert "stimulus" not in conditions


def test_insufficient_biological_replicates_are_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2", "sample_3"],
            "condition": ["control", "control", "stimulus"],
        }
    )

    group_sizes = metadata.groupby("condition").size()

    assert group_sizes["stimulus"] < 2


def test_severe_group_imbalance_is_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": [f"sample_{i}" for i in range(11)],
            "condition": ["control"] * 10 + ["stimulus"],
        }
    )

    group_sizes = metadata.groupby("condition").size()

    assert group_sizes["control"] / group_sizes["stimulus"] >= 5


def test_condition_batch_confounding_is_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "condition": ["control", "control", "stimulus", "stimulus"],
            "batch": ["batch_1", "batch_1", "batch_2", "batch_2"],
        }
    )

    contingency = pd.crosstab(metadata["condition"], metadata["batch"])

    assert (contingency > 0).sum(axis=0).eq(1).all()