"""Differential analysis between experimental conditions (Step 7).

Must report effect size + uncertainty alongside significance, and
apply multiple-testing correction (FDR) by default.

RESOLVED (project doc, section 28): this wraps PyDESeq2 — a pure-
Python, MIT-licensed re-implementation of the DESeq2 method
(Muzellec et al., Bioinformatics 2023; DOI 10.1093/bioinformatics/
btad547), rather than (a) reaching into R via rpy2, or (b)
reimplementing DESeq2's negative-binomial GLM + empirical-Bayes
dispersion shrinkage from scratch.

Why:
- rpy2 would add a full R runtime as a dependency for a Python
  framework, complicating installation, CI, and deployment, and
  introducing cross-language failure modes for no statistical
  benefit over a faithful pure-Python re-implementation.
- Reimplementing DESeq2's method from scratch (dispersion
  estimation with empirical-Bayes shrinkage, independent filtering,
  Cook's-distance outlier handling) is a large, error-prone
  undertaking. A subtly incorrect from-scratch implementation
  masquerading as a validated method is exactly the failure mode
  this project's "fail loudly rather than silently produce
  questionable analyses" and "appropriate statistical methodology"
  principles exist to prevent.
- PyDESeq2 is peer-reviewed, actively maintained, part of the
  scverse ecosystem, and tracks DESeq2's actual defaults (size-
  factor normalization, per-gene NB GLM, Wald tests, optional
  apeGLM LFC shrinkage) rather than inventing a new method — so
  "wraps an established method" is satisfied without an R
  dependency.

Consequence for preprocessing/normalization.py's contract: PyDESeq2
computes its own DESeq2 median-of-ratios size factors internally and
expects raw (un-normalized, non-negative integer) counts as input.
None of preprocessing.normalization.normalize_expression's methods
(cpm, log2, zscore) are valid inputs here — see that module's
docstring.

pydeseq2 is an optional dependency (the ``stats`` extra) rather than
a core one, so validation/QC-only installs stay lightweight. The
import is therefore deferred to inside ``compare_conditions`` itself
— importing this module must not require pydeseq2 to be installed.

Implementation shape (resolved this session):
- ``expression`` (feature-by-sample raw counts) is transposed to
  samples-by-features and wrapped in a ``DeseqDataSet`` with
  ``design="~condition"``. Only ``condition`` is used as a design
  factor — this function's signature has no covariate/batch
  parameters, so a batch-adjusted design is out of scope here (a
  natural future extension, not silently done).
- All samples in ``expression``/``metadata`` are used to fit the
  dispersion model, even if ``metadata["condition"]`` has levels
  beyond ``condition_a``/``condition_b`` — standard DESeq2 practice:
  fit the full model once, then contrast the two requested levels.
- The contrast is built as ``["condition", condition_a, condition_b]``
  (PyDESeq2/DESeq2 convention: numerator, denominator), so a positive
  ``log2_fold_change`` means higher expression in ``condition_a``.
  Verified empirically against a synthetic up-regulated gene before
  relying on it (see tests).
- Benjamini-Hochberg FDR correction and independent filtering are
  PyDESeq2's defaults and are not disabled — satisfies "apply
  multiple-testing correction (FDR) by default" without extra code.
  Independent filtering means very-low-count features can come back
  with ``NaN`` p-values/adjusted p-values; that is DESeq2 behaving as
  designed, not this function silently dropping data, and is
  documented on the return value below.
- apeGLM shrinkage is NOT applied — ``DeseqStats.summary()`` reports
  MLE log2 fold changes, not shrunken ones. Exposing shrinkage
  (``DeseqStats.lfc_shrink()``) would need its own parameter and is
  left for a future extension rather than silently changing what
  "effect size" means here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_REPLICATES_PER_CONDITION = 2

RESULT_COLUMNS = [
    "base_mean",
    "log2_fold_change",
    "lfc_se",
    "p_value",
    "p_value_adj",
]


def compare_conditions(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    """Return per-feature effect size, p-value, and adjusted p-value.

    Wraps PyDESeq2 (see module docstring for the resolved method
    decision and exact wrapping shape). ``expression`` must be a
    feature-by-sample matrix of raw, non-negative, whole-number
    counts — matching ``io.expression.load_expression_matrix``'s
    layout, and explicitly NOT the output of
    ``preprocessing.normalization.normalize_expression`` (that would
    double-normalize; PyDESeq2 computes its own size factors from raw
    counts). ``metadata`` must contain ``sample_id`` and ``condition``
    columns, matching ``io.metadata.load_metadata``'s layout.

    The contrast is ``condition_a`` vs. ``condition_b``: a positive
    ``log2_fold_change`` means higher expression in ``condition_a``.
    Any other condition levels present in ``metadata`` still
    contribute to fitting the dispersion model (standard DESeq2
    practice) but are otherwise ignored by this specific contrast.

    Returns a DataFrame indexed by feature ID (same index as
    ``expression``, same order) with:

    - ``base_mean``: mean of DESeq2-normalized counts across all
      samples used in the fit — a diagnostic, not part of the
      ``condition_a`` vs. ``condition_b`` contrast itself.
    - ``log2_fold_change``: effect size,
      ``log2(condition_a / condition_b)``.
    - ``lfc_se``: standard error of ``log2_fold_change`` — the
      uncertainty this module's contract requires alongside effect
      size.
    - ``p_value``: Wald-test p-value.
    - ``p_value_adj``: Benjamini-Hochberg FDR-adjusted p-value,
      applied automatically (never opt-in).

    Features excluded by DESeq2's independent filtering (very low
    mean count, where testing would only cost power) come back with
    ``NaN`` in ``p_value``/``p_value_adj`` — this is standard DESeq2
    behaviour surfaced as-is, not this function dropping rows.

    Raises ``ImportError`` if the optional ``pydeseq2`` dependency
    (the ``stats`` extra) is not installed. Raises ``ValueError`` if:
    ``expression`` is empty; ``condition_a`` equals ``condition_b``;
    either is absent from ``metadata["condition"]``; either has fewer
    than 2 samples (PyDESeq2 cannot estimate dispersion from a single
    replicate); the ``expression``/``metadata`` sample sets don't
    match; or ``expression`` contains negative, missing, or
    non-whole-number values.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "compare_conditions requires the optional 'pydeseq2' "
            "dependency. Install it via the 'stats' extra, e.g. "
            "`pip install stimulusbio[stats]`."
        ) from exc

    if expression.empty:
        raise ValueError(
            "Cannot run differential analysis: expression matrix is empty."
        )

    if "sample_id" not in metadata.columns or "condition" not in metadata.columns:
        raise ValueError(
            "metadata must contain 'sample_id' and 'condition' columns."
        )

    if condition_a == condition_b:
        raise ValueError(
            "condition_a and condition_b must differ; both were "
            f"'{condition_a}'."
        )

    available_conditions = set(metadata["condition"].dropna())

    for condition in (condition_a, condition_b):
        if condition not in available_conditions:
            raise ValueError(
                f"Condition '{condition}' not found in "
                f"metadata['condition']. Available conditions: "
                f"{sorted(available_conditions)}."
            )

    expression_samples = set(expression.columns)
    metadata_samples = set(metadata["sample_id"])
    missing_from_metadata = expression_samples - metadata_samples
    missing_from_expression = metadata_samples - expression_samples

    if missing_from_metadata or missing_from_expression:
        raise ValueError(
            "Expression and metadata sample sets do not match. "
            f"In expression but not metadata: {sorted(missing_from_metadata)}. "
            f"In metadata but not expression: {sorted(missing_from_expression)}."
        )

    if expression.isna().any().any():
        raise ValueError(
            "Cannot run differential analysis: expression matrix contains "
            "missing values."
        )

    expression_values = expression.to_numpy(dtype=float)

    if (expression_values < 0).any():
        raise ValueError(
            "Cannot run differential analysis: expression matrix contains "
            "negative values. compare_conditions requires raw, "
            "non-negative counts — not normalized values."
        )

    if not np.array_equal(expression_values, np.round(expression_values)):
        raise ValueError(
            "Cannot run differential analysis: expression matrix contains "
            "non-whole-number values. compare_conditions requires raw "
            "counts, not normalized or transformed values."
        )

    sample_conditions = metadata.drop_duplicates("sample_id").set_index("sample_id")[
        "condition"
    ]

    for condition in (condition_a, condition_b):
        n_replicates = int((sample_conditions == condition).sum())

        if n_replicates < MIN_REPLICATES_PER_CONDITION:
            raise ValueError(
                f"Condition '{condition}' has {n_replicates} sample(s); "
                f"PyDESeq2 requires at least {MIN_REPLICATES_PER_CONDITION} "
                "replicates per condition for dispersion estimation."
            )

    counts = expression.T.astype(int)
    counts.index.name = None

    deseq_metadata = sample_conditions.loc[counts.index].to_frame()

    dataset = DeseqDataSet(
        counts=counts,
        metadata=deseq_metadata,
        design="~condition",
        quiet=True,
    )
    dataset.deseq2()

    stats = DeseqStats(
        dataset,
        contrast=["condition", condition_a, condition_b],
        quiet=True,
    )
    stats.summary()

    result = stats.results_df.rename(
        columns={
            "baseMean": "base_mean",
            "log2FoldChange": "log2_fold_change",
            "lfcSE": "lfc_se",
            "pvalue": "p_value",
            "padj": "p_value_adj",
        }
    )[RESULT_COLUMNS]

    result = result.reindex(expression.index)
    result.index.name = expression.index.name

    return result
