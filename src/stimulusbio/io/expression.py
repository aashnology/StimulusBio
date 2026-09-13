from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_FORMATS = {
    ".csv": ",",
    ".tsv": "\t",
}


def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    """Load an expression matrix from a CSV or TSV file.

    The first column is treated as the feature identifier and becomes
    the DataFrame index. All remaining columns are preserved as sample
    columns.

    No biological transformations, filtering, normalization, or
    identifier conversion are performed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Expression matrix file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expression matrix path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Unsupported expression matrix format: '{suffix}'. "
            f"Supported formats are: {supported}."
        )

    delimiter = SUPPORTED_FORMATS[suffix]

    try:
        expression = pd.read_csv(
            path,
            sep=delimiter,
            index_col=0,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Unable to parse expression matrix: {path}"
        ) from exc

    return expression