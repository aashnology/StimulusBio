from .differential import compare_conditions
from .exploratory import compute_pca

# NOTE: importing compare_conditions here does NOT require pydeseq2 to be
# installed — the pydeseq2 import itself is deferred to inside the function
# body (see differential.py's module docstring), so validation/QC-only
# installs without the 'stats' extra can still import this package.

__all__ = [
    "compare_conditions",
    "compute_pca",
]
