"""Shared delimited-file (CSV/TSV) loading logic for the io layer.

`load_expression_matrix` and `load_metadata` both need identical
existence, file-type, and format checks, and identical error-message
shaping around `pandas.read_csv`. Centralizing that here keeps the
two loaders from drifting out of sync as new formats or checks are
added.

This module is private to `stimulusbio.io` — it defines no public
contract of its own.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_DELIMITED_FORMATS = {
    ".csv": ",",
    ".tsv": "\t",
}


def load_delimited_file(
    path: str | Path,
    label: str,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """Validate `path` and load it as a delimited (CSV/TSV) file.

    `label` (e.g. "Expression matrix", "Metadata") is interpolated
    into error messages so callers keep their existing wording.
    Extra keyword arguments are passed through to `pandas.read_csv`
    (e.g. `index_col=0` for expression matrices).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_DELIMITED_FORMATS:
        supported = ", ".join(SUPPORTED_DELIMITED_FORMATS)
        raise ValueError(
            f"Unsupported {label.lower()} format: '{suffix}'. "
            f"Supported formats are: {supported}."
        )

    delimiter = SUPPORTED_DELIMITED_FORMATS[suffix]

    try:
        return pd.read_csv(path, sep=delimiter, **read_csv_kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to parse {label.lower()}: {path}") from exc
