from __future__ import annotations

"""Compute connectome harmonics (graph Laplacian eigenpairs) from connectivity.

Pure compute utilities with minimal, explicit options and no implicit IO.

Key options exposed:
- lap_type: 'normalized' or 'combinatorial'
- k: number of smallest eigenpairs (None = full)
- adjacency prep: symmetrize, zero_diagonal, clip_negative, binarize_threshold

High-level API:
- compute_harmonics_dataset(conn_ds, ...): build an xarray Dataset with
  dims (subject_id, node, mode) and variables eigvecs/eigvals (+ optional adjacency)
  from an input xarray connectivity dataset with var 'connectivity'.


Here’s a concise breakdown of the options and what’s typical for computing of connectome harmonics

Harmonics Options

Laplacian:
normalized: symmetric normalized L = I − D^(-1/2) A D^(-1/2); most common for connectome harmonics because eigenvectors form an orthonormal basis w.r.t. the standard inner product and are less degree-biased.
combinatorial: L = D − A; classic definition, sometimes used when you want direct degree weighting.
random_walk: L = I − D^(-1) A; useful for diffusion processes; eigenvectors orthogonal in a D-weighted sense. Less common for harmonics plots but relevant for dynamics.
Spectrum size:
k: number of smallest eigenpairs; set k=None for the full basis. Typical: full for N=376 is fine; for very large N, pick k for “low-frequency” modes.
Adjacency preprocessing:
symmetrize: average with transpose; usually True for DTI connectomes.
zero_diagonal: remove self-loops; True.
clip_negative: clamp tiny negative values; True to guard numerical artifacts.
binarize_threshold: optional, converts to unweighted graph at given threshold; typically unused for DTI unless you specifically want binary adjacency.
Storage:
include_adjacency: include or omit the adjacency per subject in the harmonics dataset; typically False (store eigvals/eigvecs only) to keep file small.
Typical default for this project

lap_type='normalized', k=None, symmetrize=True, zero_diagonal=True, clip_negative=True, binarize_threshold=None, include_adjacency=False.

"""

import logging
from typing import Dict, Optional, Tuple, Mapping, Iterable

import numpy as np
import xarray as xr
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)


def _prepare_adjacency(
    A: np.ndarray,
    *,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    clip_negative: bool = True,
    binarize_threshold: float | None = None,
) -> np.ndarray:
    """Return a clean adjacency matrix for Laplacian computation.

    - symmetrize: make A = (A + A.T)/2
    - zero_diagonal: remove self-loops
    - clip_negative: clamp negatives to 0 (handles tiny rounding noise)
    - binarize_threshold: if provided, set A = 1*(A>=threshold) after cleanup
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency matrix must be square 2D array")
    if symmetrize:
        A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)
    if clip_negative:
        A[A < 0] = 0.0
    if binarize_threshold is not None:
        A = (A >= float(binarize_threshold)).astype(np.float64)
    return A


def _laplacian(A: np.ndarray, lap_type: str = "normalized") -> np.ndarray:
    """Return graph Laplacian matrix for adjacency A.

    lap_type:
      - 'combinatorial': L = D - A
      - 'normalized'  : L = I - D^{-1/2} A D^{-1/2} (symmetric normalized)
      - 'random_walk' : L = I - D^{-1} A
    """
    if lap_type == "random_walk":
        d = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            dinv = np.where(d > 0, 1.0 / d, 0.0)
        DinvA = (A * dinv[:, None])
        n = A.shape[0]
        return np.eye(n, dtype=np.float64) - DinvA
    normed = (lap_type == "normalized")
    if lap_type not in {"normalized", "combinatorial"}:
        raise ValueError("lap_type must be 'normalized', 'combinatorial', or 'random_walk'")
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
    data_dict: Mapping[int, np.ndarray],
    *,
    lap_type: str = "normalized",
    k: int | None = None,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    clip_negative: bool = True,
    binarize_threshold: float | None = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute Laplacian eigenpairs for each subject in a mapping of adjacencies.

    Returns a dict keyed by subject_id with 'eigenvectors' and 'eigenvalues'.
    """
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for sid, matrix in data_dict.items():
        A = _prepare_adjacency(
            matrix,
            symmetrize=symmetrize,
            zero_diagonal=zero_diagonal,
            clip_negative=clip_negative,
            binarize_threshold=binarize_threshold,
        )
        L = _laplacian(A, lap_type=lap_type)
        e, U = _eigendecompose_laplacian(L, k=k)
        result[sid] = {"eigenvectors": U, "eigenvalues": e}
    return result


def compute_connectome_harmonics(
    adjacency_dict: Mapping[int, np.ndarray],
    *,
    lap_type: str = "normalized",
    k: int | None = None,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    clip_negative: bool = True,
    binarize_threshold: float | None = None,
):
    """
    Returns a dictionary where each key is a subject number and each value is another dictionary with 
    keys 'eigenvectors' and 'eigenvalues' corresponding to the eigenvectors and eigenvalues of the graph Laplacian
    for the adjacency matrix associated with that subject.
    
    Parameters:
    - adjacency_dict: a dictionary where each key is a subject number and each value is a 376x376 connectivity matrix.
    - lap_type: either "combinatorial" or "normalized"
    """
    
    return compute_graph_laplacian_eigen(
        adjacency_dict,
        lap_type=lap_type,
        k=k,
        symmetrize=symmetrize,
        zero_diagonal=zero_diagonal,
        clip_negative=clip_negative,
        binarize_threshold=binarize_threshold,
    )

# Usage:
# Assuming adjacency_dict is your dictionary of adjacency matrices
# harmonics_dict = compute_harmonics(adjacency_dict)

# This function is for a numpy array rather than a dict

def compute_laplacian_eigendecomposition(
    adjacency_matrix: np.ndarray,
    *,
    lap_type: str = "normalized",
    k: int | None = None,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    clip_negative: bool = True,
    binarize_threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of the graph Laplacian of a single adjacency."""
    A = _prepare_adjacency(
        adjacency_matrix,
        symmetrize=symmetrize,
        zero_diagonal=zero_diagonal,
        clip_negative=clip_negative,
        binarize_threshold=binarize_threshold,
    )
    L = _laplacian(A, lap_type=lap_type)
    eigenvalues, eigenvectors = _eigendecompose_laplacian(L, k=k)
    return eigenvalues, eigenvectors


# -------- xarray-based API (preferred) --------

def _subject_dataset(
    subject_id: int,
    A: np.ndarray,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    *,
    atlas: Optional[str] = None,
    age: Optional[float] = None,
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
            adjacency=(("subject_id", "node", "node_2"), A[np.newaxis, :, :]),
            eigvecs=(("subject_id", "node", "mode"), eigvecs[np.newaxis, :, :]),
            eigvals=(("subject_id", "mode"), eigvals[np.newaxis, :]),
        ),
        coords=dict(
            subject_id=("subject_id", np.atleast_1d(subject_id)),
            node=("node", np.arange(n_nodes, dtype=np.int32)),
            node_2=("node_2", np.arange(n_nodes, dtype=np.int32)),
            mode=("mode", np.arange(n_modes, dtype=np.int32)),
        ),
        attrs={
            "atlas": atlas or "unknown",
            "layout": "subject_id,node,mode",
        },
    )
    if age is not None:
        ds = ds.assign_coords(age=("subject_id", np.atleast_1d(float(age))))
    return ds


def compute_subject_harmonics(
    subject_id: int,
    adjacency_matrix: np.ndarray,
    *,
    atlas: Optional[str] = None,
    lap_type: str = "normalized",
    k: Optional[int] = None,
    include_adjacency: bool = True,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    clip_negative: bool = True,
    binarize_threshold: float | None = None,
    age: Optional[float] = None,
) -> xr.Dataset:
    """Compute harmonics for a single subject and return an xarray Dataset row.

    No implicit IO. Pass the adjacency matrix explicitly. Optional age becomes a
    coordinate along 'subject_id'.
    """
    A = _prepare_adjacency(
        adjacency_matrix,
        symmetrize=symmetrize,
        zero_diagonal=zero_diagonal,
        clip_negative=clip_negative,
        binarize_threshold=binarize_threshold,
    )
    if atlas is None:
        atlas = "AAL-376" if A.shape[0] == 376 else f"n{A.shape[0]}"
    L = _laplacian(A, lap_type=lap_type)
    e, U = _eigendecompose_laplacian(L, k=k)

    # Optionally drop adjacency to reduce size
    A_to_store = A if include_adjacency else np.full_like(A, np.nan, dtype=np.float64)
    return _subject_dataset(subject_id=subject_id, A=A_to_store, eigvecs=U, eigvals=e, atlas=atlas, age=age)


def compute_harmonics_dataset(
    conn_ds: xr.Dataset,
    *,
    lap_type: str = "normalized",
    k: Optional[int] = None,
    include_adjacency: bool = False,
    ages: Optional[Mapping[int, float]] = None,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    clip_negative: bool = True,
    binarize_threshold: float | None = None,
) -> xr.Dataset:
    """Compute harmonics for all subjects in a connectivity xarray Dataset.

    Expects a variable 'connectivity' with dims (subject_id|subject, node, node_2).
    Returns a Dataset with dims (subject_id, node, mode) and variables eigvecs/eigvals
    (and optionally adjacency). If ages are provided (mapping subject_id→age),
    they are attached as a coordinate along 'subject_id'.
    """
    if "connectivity" not in conn_ds.data_vars:
        raise KeyError("Expected variable 'connectivity' in the input Dataset")

    # Normalize primary key dim to 'subject_id'
    if "subject_id" in conn_ds.dims:
        ds = conn_ds
    elif "subject" in conn_ds.dims:
        ds = conn_ds.rename({"subject": "subject_id"})
    else:
        raise KeyError("Connectivity dataset must have a 'subject' or 'subject_id' dimension")

    atlas = ds.attrs.get("atlas", "unknown")
    subject_ids: Iterable[int] = [int(s) for s in ds["subject_id"].values]

    rows: list[xr.Dataset] = []
    for sid in subject_ids:
        A = np.asarray(ds["connectivity"].sel(subject_id=sid).values, dtype=np.float64)
        age = None
        if ages is not None:
            try:
                age = float(ages[int(sid)])
            except Exception:
                age = None
        row = compute_subject_harmonics(
            subject_id=sid,
            adjacency_matrix=A,
            atlas=atlas,
            lap_type=lap_type,
            k=k,
            include_adjacency=include_adjacency,
            symmetrize=symmetrize,
            zero_diagonal=zero_diagonal,
            clip_negative=clip_negative,
            binarize_threshold=binarize_threshold,
            age=age,
        )
        rows.append(row)

    ds_h = xr.concat(rows, dim="subject_id")
    return ds_h


__all__ = [
    "compute_laplacian_eigendecomposition",
    "compute_graph_laplacian_eigen",
    "compute_connectome_harmonics",
    "compute_subject_harmonics",
    "compute_harmonics_dataset",
]
