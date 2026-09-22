from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._shared import load_delimited_file

REQUIRED_METADATA_COLUMNS = {"sample_id", "condition"}


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load experimental metadata from a CSV or TSV file.

    The metadata is loaded without biological transformations.
    The sample_id column remains a regular DataFrame column.
    """
    metadata = load_delimited_file(path, label="Metadata")

    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    return metadata
