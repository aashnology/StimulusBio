"""Package-level import surface for stimulusbio.io.

expression.py and metadata.py both delegate their delimited-file
handling to the shared _shared.load_delimited_file() helper (see
test_io_shared.py) — that DRY refactor is already complete. This file
covers the one gap left against the convention every other
implemented layer (validation/, preprocessing/, statistics/) follows:
io/__init__.py re-exporting its public loaders, so callers can do
``from stimulusbio.io import load_expression_matrix, load_metadata``
instead of reaching into submodules. Worth locking in now, before a
third loader is added to this package.
"""

from pathlib import Path

from stimulusbio.io import load_expression_matrix, load_metadata


def test_load_expression_matrix_is_importable_from_package(tmp_path: Path):
    file_path = tmp_path / "expression.csv"
    file_path.write_text("gene_id,sample_1\ngene_1,10\n")

    expression = load_expression_matrix(file_path)

    assert expression.shape == (1, 1)


def test_load_metadata_is_importable_from_package(tmp_path: Path):
    file_path = tmp_path / "metadata.csv"
    file_path.write_text("sample_id,condition\nsample_1,control\n")

    metadata = load_metadata(file_path)

    assert metadata.shape == (1, 2)
