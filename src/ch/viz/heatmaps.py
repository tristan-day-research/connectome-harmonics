from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..settings import Settings
from ..data_handling.data_utils import open_connectivity_dataset


def plot_heatmap(
    settings: Settings,
    *,
    dataset: str = "connectivity",  # "connectivity" or "harmonics"
    var: str = "connectivity",      # e.g., "connectivity", "eigvecs", "eigvals"
    atlas: Optional[str] = "AAL-376",
    sel: Optional[dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot a heatmap for any 2D slice from a stored xarray dataset.

    - dataset: select which store to open ("connectivity" or "harmonics").
    - var: name of the variable in that dataset to plot.
    - sel: dict of xarray .sel() arguments to reduce to a 2D matrix.
    """
    if dataset == "connectivity":
        ds = open_connectivity_dataset(settings, atlas=atlas)
    elif dataset == "harmonics":
        from ..compute_harmonics.harmonics import open_harmonics_dataset  # lazy import
        ds = open_harmonics_dataset(settings, atlas=atlas)
    else:
        raise ValueError("dataset must be 'connectivity' or 'harmonics'")

    # Friendly error if variable not present
    if var not in ds.data_vars:
        available = ", ".join(ds.data_vars)
        raise KeyError(f"Variable '{var}' not found in {dataset} dataset. Available: {available}")
    da = ds[var]
    if sel:
        da = da.sel(**sel)
    A = da.values
    if A.ndim != 2:
        raise ValueError(f"Selected data is not 2D; got shape {A.shape}. Provide 'sel' to pick a 2D slice.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(A, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
    ax.set_xlabel(da.dims[-1])
    ax.set_ylabel(da.dims[-2])
    if title is None:
        title = f"{dataset}:{var} {sel or ''}"
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# No wrapper needed; use plot_heatmap directly for all matrices.
