from __future__ import annotations

import pandas as pd

from .design import (
    validate_batch_confounding,
    validate_group_balance,
    validate_groups,
)
from .expression import (
    validate_expression_matrix,
    validate_metadata,
    validate_sample_alignment,
)


def validate_dataset(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    min_replicates: int = 2,
    imbalance_ratio: float = 5.0,
) -> dict[str, object]:
    """Run all available validation checks on an expression dataset."""

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(validate_expression_matrix(expression))
    errors.extend(validate_metadata(metadata))
    errors.extend(validate_sample_alignment(expression, metadata))

    warnings.extend(
        validate_groups(
            metadata,
            min_replicates=min_replicates,
        )
    )

    warnings.extend(
        validate_group_balance(
            metadata,
            imbalance_ratio=imbalance_ratio,
        )
    )

    warnings.extend(validate_batch_confounding(metadata))

    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "warnings": warnings,
    }