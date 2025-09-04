from __future__ import annotations

from typing import Callable
import numpy as np


def select_nodes(evec: np.ndarray, mode: str = "std_band", qty: float | int = 1) -> np.ndarray:
    """Return a boolean mask over nodes for a single harmonic eigenvector.

    Parameters
    - evec: shape (N,) eigenvector values for one harmonic.
    - mode: selection strategy: "std_band" | "topk" | "zscore_thresh" | "none".
    - qty: meaning depends on mode (std multiples, k count, or z-threshold).
    """
    v = np.asarray(evec).ravel()
    n = v.size

    if mode == "none":
        return np.ones(n, dtype=bool)

    if mode == "topk":
        k = int(qty)
        if k <= 0:
            return np.zeros(n, dtype=bool)
        # rank by absolute magnitude
        idx = np.argpartition(np.abs(v), -k)[-k:]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask

    # z-scores
    mu = float(np.mean(v))
    sd = float(np.std(v)) or 1.0
    z = (v - mu) / sd

    if mode == "std_band":
        t = float(qty)
        return np.abs(z) <= t

    if mode == "zscore_thresh":
        t = float(qty)
        return np.abs(z) >= t

    raise ValueError(f"Unknown selector mode: {mode}")


SelectorFn = Callable[[np.ndarray], np.ndarray]

