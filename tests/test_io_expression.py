
from pathlib import Path

import pandas as pd
import pytest

from stimulusbio.io.expression import load_expression_matrix


def test_load_expression_csv(tmp_path: Path):
    file_path = tmp_path / "expression.csv"

    file_path.write_text(
        "gene_id,sample_1,sample_2\n"
        "gene_1,10,20\n"
        "gene_2,30,40\n"
    )

    expression = load_expression_matrix(file_path)

    assert isinstance(expression, pd.DataFrame)
    assert expression.shape == (2, 2)


def test_load_expression_tsv(tmp_path: Path):
    file_path = tmp_path / "expression.tsv"

    file_path.write_text(
        "gene_id\tsample_1\tsample_2\n"
        "gene_1\t10\t20\n"
        "gene_2\t30\t40\n"
    )

    expression = load_expression_matrix(file_path)

    assert isinstance(expression, pd.DataFrame)
    assert expression.shape == (2, 2)


def test_feature_ids_are_preserved_as_index(tmp_path: Path):
    file_path = tmp_path / "expression.csv"

    file_path.write_text(
        "gene_id,sample_1,sample_2\n"
        "gene_001,10,20\n"
        "gene_002,30,40\n"
    )

    expression = load_expression_matrix(file_path)

    assert list(expression.index) == ["gene_001", "gene_002"]


def test_sample_ids_are_preserved_as_columns(tmp_path: Path):
    file_path = tmp_path / "expression.csv"

    file_path.write_text(
        "gene_id,Sample_A,Sample_B\n"
        "gene_001,10,20\n"
        "gene_002,30,40\n"
    )

    expression = load_expression_matrix(file_path)

    assert list(expression.columns) == ["Sample_A", "Sample_B"]


def test_values_are_unchanged(tmp_path: Path):
    file_path = tmp_path / "expression.csv"

    file_path.write_text(
        "gene_id,sample_1,sample_2\n"
        "gene_1,10,20\n"
        "gene_2,30,40\n"
    )

    expression = load_expression_matrix(file_path)

    expected = pd.DataFrame(
        {
            "sample_1": [10, 30],
            "sample_2": [20, 40],
        },
        index=pd.Index(["gene_1", "gene_2"], name="gene_id"),
    )

    pd.testing.assert_frame_equal(expression, expected)


def test_unsupported_file_format_is_rejected(tmp_path: Path):
    file_path = tmp_path / "expression.xlsx"
    file_path.write_text("not a supported format")

    with pytest.raises(
        ValueError,
        match="Unsupported expression matrix format",
    ):
        load_expression_matrix(file_path)


def test_missing_file_is_rejected(tmp_path: Path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Expression matrix file not found",
    ):
        load_expression_matrix(file_path)


def test_expression_loader_accepts_path_object(tmp_path: Path):
    file_path = tmp_path / "expression.csv"

    file_path.write_text(
        "gene_id,sample_1,sample_2\n"
        "gene_1,10,20\n"
        "gene_2,30,40\n"
    )

    expression = load_expression_matrix(Path(file_path))

    assert expression.shape == (2, 2)


def test_expression_loader_accepts_uppercase_extension(tmp_path: Path):
    file_path = tmp_path / "expression.CSV"

    file_path.write_text(
        "gene_id,sample_1\n"
        "gene_1,10\n"
    )

    expression = load_expression_matrix(file_path)

    assert expression.shape == (1, 1)


def test_expression_loader_preserves_row_order_and_values(tmp_path: Path):
    file_path = tmp_path / "expression.csv"

    file_path.write_text(
        "gene_id,sample_1\n"
        "gene_3,300\n"
        "gene_1,100\n"
        "gene_2,200\n"
    )

    expression = load_expression_matrix(file_path)

    assert list(expression.index) == [
        "gene_3",
        "gene_1",
        "gene_2",
    ]

    assert list(expression["sample_1"]) == [
        300,
        100,
        200,
    ]


def test_expression_loader_reports_missing_file_consistently(
    tmp_path: Path,
):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Expression matrix file not found",
    ):
        load_expression_matrix(file_path)

def test_directory_path_raises_value_error(tmp_path: Path):
    directory = tmp_path / "expression_directory"
    directory.mkdir()

    with pytest.raises(ValueError, match="path is not a file"):
        load_expression_matrix(directory)

