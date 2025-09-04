from __future__ import annotations

from typing import Iterable, Literal
import numpy as np


def per_subject(
    adj: np.ndarray,
    evecs: np.ndarray,
    *,
    builder,
    selector,
    stat: Literal["mean", "sum", "median"] = "mean",
) -> np.ndarray:
    """Compute per-subject inter-harmonic matrix using provided builder/selector.

    `builder` is typically `inter_harmonics_build.inter_harmonic_matrix`.
    """
    return builder(adj=adj, evecs=evecs, selector=selector, stat=stat)


def group(mats: Iterable[np.ndarray], method: Literal["mean", "median"] = "mean") -> np.ndarray:
    """Aggregate a list of MxM matrices into a single MxM matrix."""
    arr = [np.asarray(m) for m in mats]
    if not arr:
        raise ValueError("Empty list of matrices")
    stacked = np.stack(arr, axis=0)  # (S,M,M)
    if method == "mean":
        return np.nanmean(stacked, axis=0)
    if method == "median":
        return np.nanmedian(stacked, axis=0)
    raise ValueError(f"Unknown method: {method}")

