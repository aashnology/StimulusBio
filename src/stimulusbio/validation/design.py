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

    # Categorical columns can carry unused categories with a count of 0;
    # those aren't real experimental groups and must not be treated as one
    # (they would otherwise mask a true "fewer than two conditions" issue
    # and produce a spurious "0 sample(s)" warning).
    group_sizes = group_sizes[group_sizes > 0]

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

    # Exclude unused categorical categories (count 0): they aren't real
    # groups, and leaving them in would otherwise mask genuine imbalance
    # between the groups that do have samples.
    group_sizes = group_sizes[group_sizes > 0]

    if len(group_sizes) < 2:
        return issues

    largest_group = group_sizes.max()
    smallest_group = group_sizes.min()

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