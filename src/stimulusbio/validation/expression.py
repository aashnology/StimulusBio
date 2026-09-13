from __future__ import annotations

import pandas as pd


REQUIRED_METADATA_COLUMNS = {"sample_id", "condition"}


def validate_expression_matrix(expression: pd.DataFrame) -> list[str]:
    """Validate the structural integrity of an expression matrix."""
    issues: list[str] = []

    if expression.empty:
        issues.append("Expression matrix is empty.")
        return issues

    if expression.index.has_duplicates:
        issues.append("Expression matrix contains duplicate feature IDs.")

    if expression.columns.has_duplicates:
        issues.append("Expression matrix contains duplicate sample IDs.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression.dtypes):
        issues.append("Expression matrix contains non-numeric values.")

    if expression.isna().any().any():
        issues.append("Expression matrix contains missing values.")

    return issues


def validate_metadata(metadata: pd.DataFrame) -> list[str]:
    """Validate required experimental metadata."""
    issues: list[str] = []

    if metadata.empty:
        issues.append("Metadata table is empty.")
        return issues

    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)

    if missing_columns:
        issues.append(
            f"Metadata is missing required columns: {sorted(missing_columns)}."
        )

    if "sample_id" in metadata.columns:
        if metadata["sample_id"].duplicated().any():
            issues.append("Metadata contains duplicate sample IDs.")

    if "condition" in metadata.columns:
        if metadata["condition"].isna().any():
            issues.append("Metadata contains missing condition labels.")

    return issues


def validate_sample_alignment(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
) -> list[str]:
    """Check that expression samples and metadata samples are aligned."""
    issues: list[str] = []

    if "sample_id" not in metadata.columns:
        issues.append("Cannot check sample alignment: 'sample_id' is missing.")
        return issues

    expression_samples = set(expression.columns)
    metadata_samples = set(metadata["sample_id"])

    missing_from_metadata = expression_samples - metadata_samples
    missing_from_expression = metadata_samples - expression_samples

    if missing_from_metadata:
        issues.append(
            "Expression samples missing from metadata: "
            f"{sorted(missing_from_metadata)}."
        )

    if missing_from_expression:
        issues.append(
            "Metadata samples missing from expression matrix: "
            f"{sorted(missing_from_expression)}."
        )

    return issues