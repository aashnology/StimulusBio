from pathlib import Path

import pandas as pd
import pytest

from stimulusbio.io._shared import load_delimited_file


def test_label_is_interpolated_into_not_found_message(tmp_path: Path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Widget file not found"):
        load_delimited_file(file_path, label="Widget")


def test_label_is_interpolated_into_not_a_file_message(tmp_path: Path):
    directory = tmp_path / "widget_directory"
    directory.mkdir()

    with pytest.raises(ValueError, match="Widget path is not a file"):
        load_delimited_file(directory, label="Widget")


def test_label_is_lowercased_in_unsupported_format_message(tmp_path: Path):
    file_path = tmp_path / "widget.xlsx"
    file_path.write_text("not a supported format")

    with pytest.raises(ValueError, match="Unsupported widget format"):
        load_delimited_file(file_path, label="Widget")


def test_label_is_lowercased_in_parse_error_message(tmp_path: Path):
    file_path = tmp_path / "widget.csv"
    file_path.write_text("")

    with pytest.raises(ValueError, match="Unable to parse widget"):
        load_delimited_file(file_path, label="Widget")


def test_read_csv_kwargs_are_forwarded(tmp_path: Path):
    file_path = tmp_path / "widget.csv"
    file_path.write_text("id,a,b\nrow_1,1,2\n")

    result = load_delimited_file(file_path, label="Widget", index_col=0)

    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["row_1"]


def test_tsv_delimiter_is_applied(tmp_path: Path):
    file_path = tmp_path / "widget.tsv"
    file_path.write_text("a\tb\n1\t2\n")

    result = load_delimited_file(file_path, label="Widget")

    assert list(result.columns) == ["a", "b"]
