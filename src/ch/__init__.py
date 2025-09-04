"""Convenient imports for the connectome-harmonics package (ch).

Only this top-level __init__ is used for re-exports to keep subfolders simple.
"""

from . import settings as settings  # noqa: F401

# Analysis: inter-harmonics
from .analysis.inter_harmonics_selectors import select_nodes as ih_select_nodes  # noqa: F401
from .analysis.inter_harmonics_build import (  # noqa: F401
    summarize_block as ih_summarize_block,
    inter_harmonic_matrix as ih_matrix,
)
from .analysis.inter_harmonics_aggregate import (  # noqa: F401
    per_subject as ih_per_subject,
    group as ih_group,
)

