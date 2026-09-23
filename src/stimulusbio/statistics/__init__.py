from .exploratory import compute_pca

# NOTE: differential.compare_conditions is deliberately NOT exported here.
# It is still a stub (raises NotImplementedError) pending the open decision
# documented in differential.py's module docstring (project doc, section 28):
# whether it wraps an established DE method (e.g. rpy2 -> DESeq2/edgeR) or
# reimplements DE statistics from scratch. That decision needs its own
# dedicated chat and is not resolved here.

__all__ = [
    "compute_pca",
]
