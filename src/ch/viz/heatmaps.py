from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ..settings import Settings


def plot_heatmap(
    settings: Settings,
    *,
    path: Union[str, Path],
    subject_id: int,
    n_harmonics: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot a simple heatmap from a connectivity or harmonics file.

    - If the file has `connectivity(subject_id, node, node_2)`, plot that 376×376 matrix.
    - If the file has `eigvecs(subject_id, node, mode)`, plot a 376×k heatmap formed by
      stacking the first k eigenvectors as columns (k = n_harmonics or all if None).
    - Title includes subject_id, age (if present as a coord), and atlas (dataset) from attrs.
    """
    ds = xr.open_dataset(str(path), engine="scipy")
    atlas = ds.attrs.get("atlas", "unknown")

    # Determine subject coord name
    subj_dim = "subject_id" if "subject_id" in ds.dims else ("subject" if "subject" in ds.dims else None)
    if subj_dim is None:
        raise ValueError("Dataset must have a 'subject_id' or 'subject' dimension")
    if subject_id not in ds[subj_dim].values:
        raise ValueError(f"subject_id {subject_id} not found in dataset")

    # Age if present
    age_str = ""
    if "age" in ds.coords and subj_dim in ds["age"].dims:
        try:
            age_val = float(ds["age"].sel({subj_dim: subject_id}).values)
            age_str = f" • age={age_val:.0f}"
        except Exception:
            age_str = ""

    fig, ax = plt.subplots(figsize=figsize)

    if "connectivity" in ds.data_vars:
        A = ds["connectivity"].sel({subj_dim: subject_id}).values
        if A.ndim != 2:
            raise ValueError(f"Connectivity selection not 2D; got {A.shape}")
        im = ax.imshow(A, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_xlabel("node_2"); ax.set_ylabel("node")
        ax.set_title(f"connectivity 376×376 • subject_id={subject_id}{age_str} • atlas={atlas}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    if "eigvecs" in ds.data_vars:
        V = ds["eigvecs"].sel({subj_dim: subject_id}).values  # expect (node, mode)
        V = np.squeeze(V)
        if V.ndim != 2:
            raise ValueError(f"eigvecs expected 2D after selecting subject; got {V.shape}")
        n_nodes, n_modes = V.shape
        k = n_modes if (n_harmonics is None or n_harmonics > n_modes) else int(n_harmonics)
        V2 = V[:, :k]
        im = ax.imshow(V2, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_xlabel("mode"); ax.set_ylabel("node")
        ax.set_title(f"eigvecs {n_nodes}×{k} • subject_id={subject_id}{age_str} • atlas={atlas}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    available = ", ".join(ds.data_vars)
    raise KeyError(f"Dataset has neither 'connectivity' nor 'eigvecs'. Available: {available}")
