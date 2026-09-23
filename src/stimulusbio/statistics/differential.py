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
a core one, so validation/QC-only installs stay lightweight.

Still open, deliberately left for the implementation session: exact
wrapping shape (DeseqDataSet/DeseqStats construction from
``expression``/``metadata``), which contrast/design-formula options
to expose, and how apeGLM shrinkage is surfaced (if at all) in the
returned effect sizes. ``compare_conditions`` therefore still raises
``NotImplementedError`` — the *decision* is resolved, the
*implementation* is the next dedicated session's job, so it gets the
attention a statistical-inference layer deserves rather than being
bolted on here.
"""

from __future__ import annotations

import pandas as pd


def compare_conditions(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    """Return per-feature effect size, p-value, and adjusted p-value.

    ``expression`` must contain raw (un-normalized, non-negative
    integer) counts — see the module docstring for why. Not yet
    implemented; see module docstring for the resolved design
    decision (wraps PyDESeq2) and what remains open for the next
    implementation session.
    """
    raise NotImplementedError
