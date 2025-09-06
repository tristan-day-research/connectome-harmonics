from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging
import numpy as np
import xarray as xr
import scipy.io

from ..settings import Settings

logger = logging.getLogger(__name__)


def _infer_atlas_name(n_nodes: int) -> str:
    """Return a simple atlas name from node count.

    Keeps naming consistent across datasets while remaining generic.
    """
    # Common CamCAN export uses 376 regions; otherwise fall back to n{N}
    return "AAL-376" if n_nodes == 376 else f"n{n_nodes}"


def _build_connectivity_dataset(nets: np.ndarray, subject_ids: np.ndarray, *, atlas: Optional[str] = None) -> xr.Dataset:
    """Create an xarray Dataset for connectivity matrices.

    Dims and vars align with harmonics xarray conventions:
    - dims: subject, node, node_2
    - var:  connectivity(subject, node, node_2)
    - coords: subject (subject_id), node (0..N-1), node_2 (0..N-1)
    """
    if nets.ndim != 3:
        raise ValueError("'nets' must be a 3D array of connectivity matrices")

    # Expect nets as (N,N,S) here; ensure first two dims are equal
    if nets.shape[0] != nets.shape[1]:
        raise ValueError("'nets' must be (N,N,S) with first two dims equal (nodes)")
    n_nodes, _, n_subjects = nets.shape
    if len(subject_ids) != n_subjects:
        raise ValueError("subject_ids length must match nets' subject count")

    atlas_name = atlas or _infer_atlas_name(n_nodes)

    ds = xr.Dataset(
        data_vars=dict(
            connectivity=(("subject", "node", "node_2"), nets.transpose(2, 0, 1)),
        ),
        coords=dict(
            subject=("subject", subject_ids.astype(np.int32)),
            node=("node", np.arange(n_nodes, dtype=np.int32)),
            node_2=("node_2", np.arange(n_nodes, dtype=np.int32)),
        ),
        attrs={
            "atlas": atlas_name,
            "description": "DTI structural connectivity (fiber density)",
            "layout": "subject,node,mode-compatible",
        },
    )
    return ds


def populate_connectivity_store(settings: Settings, *, overwrite: bool = False) -> Path:
    """Load raw CamCAN connectivity and save an xarray store under processed/.

    - Reads `nets` and `age` from the MATLAB file at `settings.camcan_raw`.
    - Builds an xarray Dataset with variable `connectivity(subject,node,node_2)`.
    - Saves to `data/processed/connectivity_<atlas>.zarr` and returns the path.

    Notes on terminology: In diffusion MRI, these are typically called
    "structural connectivity" matrices (fiber density between regions).
    "Adjacency matrix" is the graph-theory term for the same numeric object.
    """
    mat_path = settings.camcan_raw
    if not mat_path.exists():
        raise FileNotFoundError(f"Raw MATLAB file not found: {mat_path}")

    # Load minimal variables from MAT file
    logger.info("Loading raw MATLAB data (keys: 'nets', 'age')")
    data = scipy.io.loadmat(str(mat_path))
    if "nets" not in data:
        raise KeyError("'nets' key not found in MATLAB file")
    if "age" not in data:
        logger.warning("'age' not found in MATLAB file; proceeding without it")

    raw_nets = np.asarray(data["nets"], dtype=np.float64)

    def _normalize_nets_shape(arr: np.ndarray) -> np.ndarray:
        """Return nets in (N,N,S) order regardless of input (N,N,S) or (S,N,N).

        Chooses the two equal-sized axes as node dims, remaining as subject dim.
        """
        if arr.ndim != 3:
            raise ValueError("Expected 3D array for 'nets'")
        a0, a1, a2 = arr.shape
        # Identify node axes as any pair of equal sizes
        pairs = []
        if a0 == a1: pairs.append((0, 1, 2))  # nodes=(0,1), subj=2
        if a0 == a2: pairs.append((0, 2, 1))  # nodes=(0,2), subj=1
        if a1 == a2: pairs.append((1, 2, 0))  # nodes=(1,2), subj=0
        if not pairs:
            raise ValueError(f"Cannot infer node axes from shape {arr.shape}; two axes must match")
        # Prefer nodes=(0,1) if available, else take the first candidate
        nodes_i, nodes_j, subj_axis = (0, 1, 2) if (0, 1, 2) in pairs else pairs[0]
        # If nodes axes are already (0,1), we only need to ensure subject is last
        if (nodes_i, nodes_j, subj_axis) == (0, 1, 2):
            return arr  # already (N,N,S)
        # Build transpose order to (nodes_i, nodes_j, subj_axis) → (0,1,2)
        # We want resulting order (0,1,2) = (nodes_i, nodes_j, subj_axis)
        transpose_order = [nodes_i, nodes_j, subj_axis]
        # Now bring to canonical order (0,1,2)
        inv = np.argsort(transpose_order)
        return np.transpose(arr, axes=transpose_order)[...,] if inv.tolist() == [0,1,2] else np.transpose(arr, axes=transpose_order)

    nets_nns = _normalize_nets_shape(raw_nets)
    n_nodes, _, n_subjects = nets_nns.shape
    subject_ids = np.arange(1, n_subjects + 1, dtype=np.int32)

    ds = _build_connectivity_dataset(nets_nns, subject_ids)

    # Decide store path by atlas
    atlas = ds.attrs.get("atlas", _infer_atlas_name(n_nodes))
    store = settings.processed_dir / f"connectivity_{atlas}.nc"
    store.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing dataset
    if store.exists() and not overwrite:
        msg = (
            "WARNING: xarray for connectivity matrices for this dataset already exist! "
            "To create another delete the current one."
        )
        raise FileExistsError(msg)

    # Use the built-in scipy engine to avoid extra deps (netCDF3 format)
    ds.to_netcdf(str(store), engine="scipy")

    rel = store.relative_to(settings.data_root) if store.is_absolute() else store
    logger.info(f"Connectivity Zarr written at {rel}")
    return store


def open_connectivity_dataset(settings: Settings, atlas: Optional[str] = None) -> xr.Dataset:
    """Open the saved connectivity store as an xarray Dataset.

    If `atlas` is None, attempts the default CamCAN atlas naming based on node count.
    """
    if atlas is None:
        # Best-effort default name for CamCAN
        atlas = "AAL-376"
    store = settings.processed_dir / f"connectivity_{atlas}.nc"
    if not store.exists():
        raise FileNotFoundError(f"Connectivity store not found: {store}")
    return xr.open_dataset(store, engine="scipy")
