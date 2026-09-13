import pandas as pd

from stimulusbio.validation import validate_dataset
from stimulusbio.validation.design import (
    validate_batch_confounding,
    validate_group_balance,
    validate_groups,
)
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

    issues = validate_groups(metadata)

    assert "Experimental design contains fewer than two conditions." in issues


def test_insufficient_biological_replicates_are_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2", "sample_3"],
            "condition": ["control", "control", "stimulus"],
        }
    )

    issues = validate_groups(metadata, min_replicates=2)

    assert any("stimulus" in issue for issue in issues)
    assert any("biological replicates" in issue for issue in issues)


def test_severe_group_imbalance_is_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": [f"sample_{i}" for i in range(11)],
            "condition": ["control"] * 10 + ["stimulus"],
        }
    )

    issues = validate_group_balance(metadata)

    assert any("Severe group imbalance" in issue for issue in issues)


def test_condition_batch_confounding_is_detected():
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "condition": ["control", "control", "stimulus", "stimulus"],
            "batch": ["batch_1", "batch_1", "batch_2", "batch_2"],
        }
    )

    issues = validate_batch_confounding(metadata)

    assert any("confounded" in issue for issue in issues)


# -------------------------
# Unified dataset validation
# -------------------------

def test_validate_dataset_passes_valid_dataset():
    expression = pd.DataFrame(
        {
            "sample_1": [10, 20, 30],
            "sample_2": [11, 21, 31],
            "sample_3": [12, 22, 32],
            "sample_4": [13, 23, 33],
        },
        index=["gene_1", "gene_2", "gene_3"],
    )

    metadata = pd.DataFrame(
        {
            "sample_id": [
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
            ],
            "condition": [
                "control",
                "control",
                "stimulus",
                "stimulus",
            ],
        }
    )

    result = validate_dataset(expression, metadata)

    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_validate_dataset_fails_invalid_expression_data():
    expression = pd.DataFrame(
        {
            "sample_1": [10, None],
            "sample_2": [11, 21],
        },
        index=["gene_1", "gene_2"],
    )

    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "condition": ["control", "stimulus"],
        }
    )

    result = validate_dataset(expression, metadata)

    assert result["status"] == "FAIL"
    assert "Expression matrix contains missing values." in result["errors"]


def test_validate_dataset_reports_design_warnings():
    expression = pd.DataFrame(
        {
            "sample_1": [10],
            "sample_2": [20],
            "sample_3": [30],
        },
        index=["gene_1"],
    )

    metadata = pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2", "sample_3"],
            "condition": ["control", "control", "stimulus"],
        }
    )

    result = validate_dataset(expression, metadata)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert len(result["warnings"]) > 0