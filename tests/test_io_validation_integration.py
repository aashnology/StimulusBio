from pathlib import Path

import pandas as pd

from stimulusbio.io.expression import load_expression_matrix
from stimulusbio.io.metadata import load_metadata
from stimulusbio.validation import validate_dataset


def test_load_and_validate_valid_dataset(tmp_path: Path):
    expression_path = tmp_path / "expression.csv"
    metadata_path = tmp_path / "metadata.csv"

    expression_path.write_text(
        "gene_id,sample_1,sample_2,sample_3,sample_4\n"
        "gene_1,10,20,30,40\n"
        "gene_2,15,25,35,45\n"
    )

    metadata_path.write_text(
        "sample_id,condition\n"
        "sample_1,control\n"
        "sample_2,control\n"
        "sample_3,stimulus\n"
        "sample_4,stimulus\n"
    )

    expression = load_expression_matrix(expression_path)
    metadata = load_metadata(metadata_path)

    result = validate_dataset(expression, metadata)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["warnings"] == []


def test_load_and_validate_detects_sample_mismatch(tmp_path: Path):
    expression_path = tmp_path / "expression.csv"
    metadata_path = tmp_path / "metadata.csv"

    expression_path.write_text(
        "gene_id,sample_1,sample_2,sample_3\n"
        "gene_1,10,20,30\n"
        "gene_2,15,25,35\n"
    )

    metadata_path.write_text(
        "sample_id,condition\n"
        "sample_1,control\n"
        "sample_2,stimulus\n"
        "sample_4,stimulus\n"
    )

    expression = load_expression_matrix(expression_path)
    metadata = load_metadata(metadata_path)

    result = validate_dataset(expression, metadata)

    assert result["status"] == "FAIL"
    assert result["errors"] == [
        "Expression samples missing from metadata: ['sample_3'].",
        "Metadata samples missing from expression matrix: ['sample_4'].",
    ]


def test_validation_preserves_expression_data(tmp_path: Path):
    expression_path = tmp_path / "expression.csv"
    metadata_path = tmp_path / "metadata.csv"

    expression_path.write_text(
        "gene_id,sample_3,sample_1,sample_2\n"
        "gene_3,300,100,200\n"
        "gene_1,30,10,20\n"
        "gene_2,3000,1000,2000\n"
    )

    metadata_path.write_text(
        "sample_id,condition\n"
        "sample_3,stimulus\n"
        "sample_1,control\n"
        "sample_2,control\n"
    )

    expression = load_expression_matrix(expression_path)
    metadata = load_metadata(metadata_path)

    original_expression = expression.copy(deep=True)

    validate_dataset(expression, metadata)

    pd.testing.assert_frame_equal(expression, original_expression)


def test_validation_preserves_metadata_data(tmp_path: Path):
    expression_path = tmp_path / "expression.csv"
    metadata_path = tmp_path / "metadata.csv"

    expression_path.write_text(
        "gene_id,sample_3,sample_1,sample_2\n"
        "gene_3,300,100,200\n"
        "gene_1,30,10,20\n"
    )

    metadata_path.write_text(
        "sample_id,condition,batch,timepoint\n"
        "sample_3,stimulus,batch_2,48h\n"
        "sample_1,control,batch_1,24h\n"
        "sample_2,control,batch_1,24h\n"
    )

    expression = load_expression_matrix(expression_path)
    metadata = load_metadata(metadata_path)

    original_metadata = metadata.copy(deep=True)

    validate_dataset(expression, metadata)

    pd.testing.assert_frame_equal(metadata, original_metadata)