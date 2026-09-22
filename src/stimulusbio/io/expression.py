from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._shared import load_delimited_file


def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    """Load an expression matrix from a CSV or TSV file.

    The first column is treated as the feature identifier and becomes
    the DataFrame index. All remaining columns are preserved as sample
    columns.

    No biological transformations, filtering, normalization, or
    identifier conversion are performed.
    """
    return load_delimited_file(path, label="Expression matrix", index_col=0)
