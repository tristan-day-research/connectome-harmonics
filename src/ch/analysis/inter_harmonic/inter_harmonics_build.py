from __future__ import annotations

from typing import Callable, Literal
import numpy as np


def summarize_block(
    adj: np.ndarray,
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    stat: Literal["mean", "sum", "median"] = "mean",
) -> float:
    """Summarize connectivity between two node sets.

    adj: (N,N) symmetric connectivity matrix.
    mask_i, mask_j: boolean masks over nodes (N,).
    stat: aggregation.
    """
    A = np.asarray(adj)
    mi = np.asarray(mask_i, dtype=bool)
    mj = np.asarray(mask_j, dtype=bool)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj must be square (N,N)")
    if mi.shape != (A.shape[0],) or mj.shape != (A.shape[0],):
        raise ValueError("mask shapes must be (N,)")

    block = A[np.ix_(mi, mj)]
    if block.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.nanmean(block))
    if stat == "sum":
        return float(np.nansum(block))
    if stat == "median":
        return float(np.nanmedian(block))
    raise ValueError(f"Unknown stat: {stat}")


def inter_harmonic_matrix(
    adj: np.ndarray,
    evecs: np.ndarray,
    selector: Callable[[np.ndarray], np.ndarray],
    *,
    stat: Literal["mean", "sum", "median"] = "mean",
    sym: bool = True,
) -> np.ndarray:
    """Compute an MxM inter-harmonic connectivity matrix.

    Parameters
    - adj: (N,N) connectivity.
    - evecs: (N,M) eigenvectors (columns are harmonics).
    - selector: function mapping a (N,) eigenvector to a boolean mask.
    - stat: aggregation statistic for blocks.
    - sym: if True, reflect upper triangle to lower.
    """
    A = np.asarray(adj)
    V = np.asarray(evecs)
    if V.ndim != 2:
        raise ValueError("evecs must be (N,M)")
    N, M = V.shape
    if A.shape != (N, N):
        raise ValueError("adj shape must match evecs (N,N)")

    masks = [np.asarray(selector(V[:, k]), dtype=bool) for k in range(M)]
    out = np.full((M, M), np.nan, dtype=float)
    for i in range(M):
        mi = masks[i]
        for j in range(i, M):
            mj = masks[j]
            out[i, j] = summarize_block(A, mi, mj, stat=stat)
            if sym and i != j:
                out[j, i] = out[i, j]
    return out

