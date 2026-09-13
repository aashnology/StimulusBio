from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_FORMATS = {
    ".csv": ",",
    ".tsv": "\t",
}

REQUIRED_METADATA_COLUMNS = {"sample_id", "condition"}


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load experimental metadata from a CSV or TSV file.

    The metadata is loaded without biological transformations.
    The sample_id column remains a regular DataFrame column.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Metadata path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Unsupported metadata format: '{suffix}'. "
            f"Supported formats are: {supported}."
        )

    delimiter = SUPPORTED_FORMATS[suffix]

    try:
        metadata = pd.read_csv(
            path,
            sep=delimiter,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Unable to parse metadata: {path}"
        ) from exc

    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    return metadata