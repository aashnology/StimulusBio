from __future__ import annotations

import pandas as pd


def validate_groups(
    metadata: pd.DataFrame,
    min_replicates: int = 2,
) -> list[str]:
    """Validate experimental condition groups and biological replication."""
    issues: list[str] = []

    if "condition" not in metadata.columns:
        issues.append("Cannot validate groups: 'condition' is missing.")
        return issues

    conditions = metadata["condition"].dropna()

    if conditions.empty:
        issues.append("No experimental conditions are defined.")
        return issues

    group_sizes = conditions.value_counts()

    if len(group_sizes) < 2:
        issues.append("Experimental design contains fewer than two conditions.")

    for condition, size in group_sizes.items():
        if size < min_replicates:
            issues.append(
                f"Condition '{condition}' has {size} sample(s); "
                f"at least {min_replicates} biological replicates are recommended."
            )

    return issues


def validate_group_balance(
    metadata: pd.DataFrame,
    imbalance_ratio: float = 5.0,
) -> list[str]:
    """Detect severe imbalance between experimental groups."""
    issues: list[str] = []

    if "condition" not in metadata.columns:
        issues.append("Cannot validate group balance: 'condition' is missing.")
        return issues

    group_sizes = metadata["condition"].dropna().value_counts()

    if len(group_sizes) < 2:
        return issues

    largest_group = group_sizes.max()
    smallest_group = group_sizes.min()

    if smallest_group == 0:
        return issues

    ratio = largest_group / smallest_group

    if ratio >= imbalance_ratio:
        issues.append(
            f"Severe group imbalance detected: largest group is "
            f"{ratio:.1f}x the size of the smallest group."
        )

    return issues


def validate_batch_confounding(
    metadata: pd.DataFrame,
) -> list[str]:
    """Detect complete confounding between condition and batch."""
    issues: list[str] = []

    required_columns = {"condition", "batch"}

    if not required_columns.issubset(metadata.columns):
        return issues

    data = metadata[["condition", "batch"]].dropna()

    if data.empty:
        return issues

    contingency = pd.crosstab(data["condition"], data["batch"])

    # A batch containing samples from only one condition indicates
    # potential complete confounding.
    for batch in contingency.columns:
        nonzero_conditions = (contingency[batch] > 0).sum()

        if nonzero_conditions == 1:
            issues.append(
                f"Batch '{batch}' contains samples from only one condition; "
                "condition and batch may be confounded."
            )

    return issues