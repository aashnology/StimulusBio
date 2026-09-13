
from pathlib import Path

import pandas as pd
import pytest

from stimulusbio.io.metadata import load_metadata


def test_load_metadata_csv(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition\n"
        "sample_1,control\n"
        "sample_2,stimulus\n"
    )

    metadata = load_metadata(file_path)

    assert isinstance(metadata, pd.DataFrame)
    assert metadata.shape == (2, 2)


def test_load_metadata_tsv(tmp_path: Path):
    file_path = tmp_path / "metadata.tsv"

    file_path.write_text(
        "sample_id\tcondition\n"
        "sample_1\tcontrol\n"
        "sample_2\tstimulus\n"
    )

    metadata = load_metadata(file_path)

    assert isinstance(metadata, pd.DataFrame)
    assert metadata.shape == (2, 2)


def test_sample_id_is_preserved_as_column(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition\n"
        "sample_1,control\n"
        "sample_2,stimulus\n"
    )

    metadata = load_metadata(file_path)

    assert "sample_id" in metadata.columns
    assert list(metadata["sample_id"]) == ["sample_1", "sample_2"]
    assert metadata.index.tolist() == [0, 1]


def test_condition_is_preserved_as_column(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition\n"
        "sample_1,control\n"
        "sample_2,stimulus\n"
    )

    metadata = load_metadata(file_path)

    assert "condition" in metadata.columns
    assert list(metadata["condition"]) == ["control", "stimulus"]


def test_additional_columns_are_preserved(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition,batch,timepoint,cell_type\n"
        "sample_1,control,batch_1,24h,neuron\n"
        "sample_2,stimulus,batch_2,24h,neuron\n"
    )

    metadata = load_metadata(file_path)

    assert list(metadata.columns) == [
        "sample_id",
        "condition",
        "batch",
        "timepoint",
        "cell_type",
    ]


def test_row_order_is_preserved(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition\n"
        "sample_3,stimulus\n"
        "sample_1,control\n"
        "sample_2,control\n"
    )

    metadata = load_metadata(file_path)

    assert list(metadata["sample_id"]) == [
        "sample_3",
        "sample_1",
        "sample_2",
    ]


def test_values_are_unchanged(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition,batch\n"
        "Sample_A,Control,Batch_1\n"
        "Sample_B,Stimulus,Batch_2\n"
    )

    metadata = load_metadata(file_path)

    expected = pd.DataFrame(
        {
            "sample_id": ["Sample_A", "Sample_B"],
            "condition": ["Control", "Stimulus"],
            "batch": ["Batch_1", "Batch_2"],
        }
    )

    pd.testing.assert_frame_equal(metadata, expected)


def test_missing_sample_id_column_is_rejected(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "condition\n"
        "control\n"
        "stimulus\n"
    )

    with pytest.raises(ValueError, match="sample_id"):
        load_metadata(file_path)


def test_missing_condition_column_is_rejected(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id\n"
        "sample_1\n"
        "sample_2\n"
    )

    with pytest.raises(ValueError, match="condition"):
        load_metadata(file_path)


def test_unsupported_file_format_is_rejected(tmp_path: Path):
    file_path = tmp_path / "metadata.xlsx"
    file_path.write_text("not a supported format")

    with pytest.raises(ValueError, match="Unsupported metadata format"):
        load_metadata(file_path)


def test_missing_file_is_rejected(tmp_path: Path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load_metadata(file_path)


def test_empty_file_is_rejected(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"
    file_path.write_text("")

    with pytest.raises(ValueError, match="Unable to parse metadata"):
        load_metadata(file_path)


def test_metadata_loader_accepts_path_object(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition\n"
        "sample_1,control\n"
        "sample_2,stimulus\n"
    )

    metadata = load_metadata(Path(file_path))

    assert metadata.shape == (2, 2)


def test_metadata_loader_accepts_uppercase_extension(tmp_path: Path):
    file_path = tmp_path / "metadata.TSV"

    file_path.write_text(
        "sample_id\tcondition\n"
        "sample_1\tcontrol\n"
        "sample_2\tstimulus\n"
    )

    metadata = load_metadata(file_path)

    assert metadata.shape == (2, 2)


def test_metadata_loader_preserves_row_order_and_values(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"

    file_path.write_text(
        "sample_id,condition,batch\n"
        "sample_3,stimulus,batch_2\n"
        "sample_1,control,batch_1\n"
        "sample_2,control,batch_1\n"
    )

    metadata = load_metadata(file_path)

    assert list(metadata["sample_id"]) == [
        "sample_3",
        "sample_1",
        "sample_2",
    ]

    assert list(metadata["condition"]) == [
        "stimulus",
        "control",
        "control",
    ]

    assert list(metadata["batch"]) == [
        "batch_2",
        "batch_1",
        "batch_1",
    ]


def test_metadata_loader_reports_missing_file_consistently(tmp_path: Path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Metadata file not found",
    ):
        load_metadata(file_path)

