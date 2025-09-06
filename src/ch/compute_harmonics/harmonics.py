from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import pygsp

from ..data_handling.metadata import load_metadata
from ..data_handling.io import get_nets_and_ages

logger = logging.getLogger(__name__)


def _prepare_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetrize, zero the diagonal, ensure float64, and clip negatives."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency matrix must be square 2D array")
    # Symmetrize
    A = 0.5 * (A + A.T)
    # Remove self loops
    np.fill_diagonal(A, 0.0)
    # Clip tiny negative weights (e.g., due to rounding)
    A[A < 0] = 0.0
    return A


def _laplacian(A: np.ndarray, lap_type: str = "normalized") -> np.ndarray:
    normed = (lap_type == "normalized")
    if lap_type not in {"normalized", "combinatorial"}:
        raise ValueError("lap_type must be 'normalized' or 'combinatorial'")
    return laplacian(A, normed=normed)


def _eigendecompose_laplacian(L: np.ndarray, k: int | None = None):
    """Compute eigendecomposition of Laplacian L.

    - If k is None or k >= n-1, computes full dense eigendecomposition (eigh).
    - Otherwise, uses sparse eigensolver to get the k smallest eigenpairs.
    Returns (eigenvalues, eigenvectors) with eigenvalues ascending.
    """
    n = L.shape[0]
    if k is None or k >= n - 1:
        w, U = eigh(L)
        return w, U

    # Use sparse solver for k smallest magnitude (near 0 for Laplacian)
    # Ensure sparse format for efficiency
    L_sparse = csr_matrix(L) if not issparse(L) else L
    w, U = eigsh(L_sparse, k=k, which="SM")
    # eigsh does not guarantee sorted order ascending
    order = np.argsort(w)
    return w[order], U[:, order]


def compute_graph_laplacian_eigen(
    data_dict: Dict[int, np.ndarray],
    lap_type: str = "normalized",
    k: int | None = None,
):
    """Compute Laplacian eigenpairs for each subject in a dict of adjacencies.

    Returns a dict for convenience; prefer xarray functions below for persistent storage.
    """
    result_dict: Dict[int, Dict[str, np.ndarray]] = {}
    for subject, matrix in data_dict.items():
        A = _prepare_adjacency(matrix)
        L = _laplacian(A, lap_type=lap_type)
        eigenvalues, eigenvectors = _eigendecompose_laplacian(L, k=k)
        result_dict[subject] = {
            'eigenvectors': eigenvectors,
            'eigenvalues': eigenvalues,
        }
    return result_dict


def compute_connectome_harmonics(adjacency_dict, lap_type: str = "normalized", k: int | None = None):
    """
    Returns a dictionary where each key is a subject number and each value is another dictionary with 
    keys 'eigenvectors' and 'eigenvalues' corresponding to the eigenvectors and eigenvalues of the graph Laplacian
    for the adjacency matrix associated with that subject.
    
    Parameters:
    - adjacency_dict: a dictionary where each key is a subject number and each value is a 376x376 connectivity matrix.
    - lap_type: either "combinatorial" or "normalized"
    """
    
    result_dict = {}

    for subject, adjacency_matrix in adjacency_dict.items():
        # Prepare adjacency (symmetrize, zero diag, clip negatives)
        A = _prepare_adjacency(adjacency_matrix)

        # Use PyGSP for Laplacian + basis if full basis is requested
        if k is None:
            G_fd = pygsp.graphs.Graph(A)
            G_fd.compute_laplacian(lap_type=lap_type)
            G_fd.compute_fourier_basis()
            result_dict[subject] = {
                'eigenvectors': G_fd.U,
                'eigenvalues': G_fd.e,
            }
        else:
            # For partial spectra, use SciPy path for efficiency
            L = _laplacian(A, lap_type=lap_type)
            e, U = _eigendecompose_laplacian(L, k=k)
            result_dict[subject] = {
                'eigenvectors': U,
                'eigenvalues': e,
            }

    return result_dict

# Usage:
# Assuming adjacency_dict is your dictionary of adjacency matrices
# harmonics_dict = compute_harmonics(adjacency_dict)

# This function is for a numpy array rather than a dict

def compute_laplacian_eigendecomposition(adjacency_matrix, lap_type: str = "normalized", k: int | None = None):
    """
    Compute eigenvalues and eigenvectors of the graph Laplacian of a single adjacency matrix.
    
    Parameters:
    - adjacency_matrix: 2D NumPy array (square)
    - lap_type: 'normalized' or 'combinatorial'
    - k: number of smallest eigenpairs to return (None = full)
    
    Returns:
    - eigenvalues: 1D NumPy array
    - eigenvectors: 2D NumPy array (columns are eigenvectors)
    """
    A = _prepare_adjacency(adjacency_matrix)
    L = _laplacian(A, lap_type=lap_type)
    eigenvalues, eigenvectors = _eigendecompose_laplacian(L, k=k)
    return eigenvalues, eigenvectors


# -------- xarray-based API (preferred) --------

def _subject_dataset(
    subject_id: int,
    A: np.ndarray,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    atlas: Optional[str] = None,
) -> xr.Dataset:
    """Build an xarray Dataset for a single subject with dims:
    - subject: [subject_id]
    - node: [0..n_nodes-1]
    - mode: [0..n_modes-1]
    Variables: adjacency (subject,node,node), eigvecs (subject,node,mode), eigvals (subject,mode)
    """
    n_nodes = A.shape[0]
    n_modes = eigvals.shape[0]

    ds = xr.Dataset(
        data_vars=dict(
            adjacency=("subject", "node", "node_2", A[np.newaxis, :, :]),
            eigvecs=("subject", "node", "mode", eigvecs[np.newaxis, :, :]),
            eigvals=("subject", "mode", eigvals[np.newaxis, :]),
        ),
        coords=dict(
            subject=("subject", np.atleast_1d(subject_id)),
            node=("node", np.arange(n_nodes, dtype=np.int32)),
            node_2=("node_2", np.arange(n_nodes, dtype=np.int32)),
            mode=("mode", np.arange(n_modes, dtype=np.int32)),
        ),
        attrs={
            "atlas": atlas or "unknown",
            "layout": "subject,node,mode",
        },
    )
    return ds


def compute_subject_harmonics(
    settings: Settings,
    subject_id: int,
    adjacency_matrix: Optional[np.ndarray] = None,
    *,
    atlas: Optional[str] = None,
    lap_type: Optional[str] = None,
    k: Optional[int] = None,
    include_adjacency: bool = True,
) -> xr.Dataset:
    """Compute harmonics for a single subject and return an xarray Dataset.

    - Pass `adjacency_matrix` if you already have it; otherwise this will try to load
      it from the CamCAN raw `.mat` (subject_id as 1-indexed).
    - `lap_type` defaults to `settings.lap_type`.
    - If `k` is provided, computes only the k smallest eigenpairs.
    - `include_adjacency=False` will skip writing adjacency to reduce disk.
    """
    lap = lap_type or settings.lap_type

    if adjacency_matrix is None:
        # Fallback loader for CamCAN: nets[:,:,i] with 1-indexed subject_id
        if not settings.camcan_raw.exists():
            raise FileNotFoundError(
                "Adjacency not provided and camcan_raw not found. Supply adjacency_matrix."
            )
        nets, _ages = get_nets_and_ages(settings.camcan_raw)
        idx = int(subject_id) - 1
        if idx < 0 or idx >= nets.shape[-1]:
            raise IndexError(f"subject_id {subject_id} out of range for CamCAN nets")
        A = np.array(nets[:, :, idx], dtype=np.float64)
        if atlas is None:
            atlas = "AAL-376" if A.shape[0] == 376 else f"n{A.shape[0]}"
    else:
        A = np.asarray(adjacency_matrix, dtype=np.float64)
        if atlas is None:
            atlas = f"n{A.shape[0]}"

    A = _prepare_adjacency(A)
    L = _laplacian(A, lap_type=lap)
    e, U = _eigendecompose_laplacian(L, k=k)

    # Optionally drop adjacency to save space
    A_to_store = A if include_adjacency else np.empty((A.shape[0], A.shape[1]), dtype=np.float64)
    if not include_adjacency:
        A_to_store[:] = np.nan

    return _subject_dataset(subject_id=subject_id, A=A_to_store, eigvecs=U, eigvals=e, atlas=atlas)


def save_subject_to_zarr(
    ds_subject: xr.Dataset,
    store_path: str | "Path",
    *,
    append: bool = True,
) -> None:
    """Append a single-subject Dataset (with subject dim length 1) to a Zarr store.
    Creates the store if missing; otherwise appends along the 'subject' dimension.
    """
    path = str(store_path)

    # Choose simple chunking that enables appends
    n_nodes = ds_subject.dims["node"]
    n_modes = ds_subject.dims["mode"]
    encoding = {
        "adjacency": {"chunks": (1, n_nodes, n_nodes)},
        "eigvecs": {"chunks": (1, n_nodes, min(128, n_modes))},
        "eigvals": {"chunks": (1, min(128, n_modes))},
    }

    mode = "a" if append else "w"
    ds_subject.to_zarr(path, mode=mode, append_dim="subject", encoding=encoding)


def compute_and_store_harmonics_for_all_subjects(
    settings: Settings,
    *,
    atlas: Optional[str] = None,
    lap_type: Optional[str] = None,
    k: Optional[int] = None,
    include_adjacency: bool = True,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> str:
    """Compute harmonics for all subjects listed in subjects metadata and store in Zarr.

    Strategy:
    - Loads subjects from `settings.subjects_parquet` (index is `subject_id`).
    - For each subject, computes harmonics via `compute_subject_harmonics`.
    - Writes to a per-atlas Zarr dataset at `settings.harmonics_dir / harmonics_<atlas>.zarr`.
    - Appends along the `subject` dimension; use `overwrite=True` to start fresh.

    Returns the Zarr store path as a string.
    """
    # Load metadata and figure subject list
    meta = load_metadata(settings, table="subjects")
    if "subject_id" in meta.columns:
        meta = meta.set_index("subject_id")
    subject_ids = list(meta.index.astype(int))
    if limit is not None:
        subject_ids = subject_ids[: int(limit)]

    # Determine atlas name
    atlas_name = atlas or (meta["atlas"].iloc[0] if "atlas" in meta.columns and len(meta) > 0 else None)
    store = settings.harmonics_store_path(atlas=atlas_name)
    store.parent.mkdir(parents=True, exist_ok=True)

    # Fresh start if requested
    if overwrite and store.exists():
        import shutil
        shutil.rmtree(store, ignore_errors=True)

    # Iterate and append
    for i, sid in enumerate(subject_ids, start=1):
        ds_sub = compute_subject_harmonics(
            settings,
            subject_id=sid,
            adjacency_matrix=None,
            atlas=atlas_name,
            lap_type=lap_type,
            k=k,
            include_adjacency=include_adjacency,
        )
        save_subject_to_zarr(ds_sub, store, append=True)

        if i % 10 == 0 or i == len(subject_ids):
            logger.info(f"Computed harmonics for {i}/{len(subject_ids)} subjects")

    rel = store.relative_to(settings.data_root) if store.is_absolute() else store
    logger.info(f"Harmonics Zarr saved/appended at {rel}")
    return str(store)


def open_harmonics_dataset(settings: Settings, atlas: Optional[str] = None) -> xr.Dataset:
    """Open the harmonics Zarr dataset for an atlas as an xarray Dataset."""
    store = settings.harmonics_store_path(atlas=atlas)
    if not store.exists():
        raise FileNotFoundError(f"Harmonics store not found: {store}")
    return xr.open_zarr(store)
