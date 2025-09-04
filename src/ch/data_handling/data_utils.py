
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Union, Optional


def load_metadata(metadata_path: Union[str, Path]) -> pd.DataFrame:
    """Load subject metadata from parquet file.
    
    Parameters
    ----------
    metadata_path : str or Path
        Path to the metadata parquet file
        
    Returns
    -------
    pd.DataFrame
        Subject metadata with subject_id as index
    """
    return pd.read_parquet(metadata_path)


def load_connectivity(connectivity_path: Union[str, Path]) -> xr.DataArray:
    """Load connectivity matrices from parquet file.
    
    Parameters
    ----------
    connectivity_path : str or Path
        Path to the connectivity parquet file
        
    Returns
    -------
    xr.DataArray
        Connectivity matrices with dimensions (subject_id, region_i, region_j)
    """
    # Load the parquet file
    df = pd.read_parquet(connectivity_path)
    
    # Convert to xarray DataArray
    # Assuming the parquet has columns: subject_id, region_i, region_j, connectivity_value
    da = df.set_index(['subject_id', 'region_i', 'region_j'])['connectivity_value'].to_xarray()
    
    return da


def load_connectivity_simple(connectivity_path: Union[str, Path]) -> np.ndarray:
    """Load connectivity matrices as simple numpy array.
    
    Parameters
    ----------
    connectivity_path : str or Path
        Path to the connectivity parquet file
        
    Returns
    -------
    np.ndarray
        Connectivity matrices with shape (n_subjects, n_regions, n_regions)
    """
    # Load the parquet file
    df = pd.read_parquet(connectivity_path)
    
    # Get dimensions
    n_subjects = df['subject_id'].nunique()
    n_regions = df['region_i'].nunique()
    
    # Create array
    connectivity = np.zeros((n_subjects, n_regions, n_regions))
    
    # Fill array
    for _, row in df.iterrows():
        subj_idx = row['subject_id'] - 1  # Convert to 0-based indexing
        i_idx = row['region_i']
        j_idx = row['region_j']
        connectivity[subj_idx, i_idx, j_idx] = row['connectivity_value']
    
    return connectivity


def load_harmonics(harmonics_path: Union[str, Path]) -> xr.DataArray:
    """Load connectome harmonics from parquet file.
    
    Parameters
    ----------
    harmonics_path : str or Path
        Path to the harmonics parquet file
        
    Returns
    -------
    xr.DataArray
        Harmonics dataset
    """
    df = pd.read_parquet(harmonics_path)
    da = df.set_index(['subject_id', 'harmonic', 'region'])['harmonic_value'].to_xarray()
    return da


def load_yeo_labels(yeo_path: Union[str, Path]):
    """Load yeo labels from pickle file.
    
    Parameters
    ----------
    yeo_path : str or Path
        Path to the yeo labels pickle file
        
    Returns
    -------
    Any
        Yeo labels data
    """
    import pickle
    with open(yeo_path, 'rb') as f:
        return pickle.load(f)


def get_subject_connectivity(connectivity_data, subject_id: int) -> np.ndarray:
    """Get connectivity matrix for a specific subject.
    
    Parameters
    ----------
    connectivity_data : xr.DataArray or np.ndarray
        Connectivity data
    subject_id : int
        Subject ID (1-based)
        
    Returns
    -------
    np.ndarray
        Connectivity matrix for the subject
    """
    if isinstance(connectivity_data, xr.DataArray):
        return connectivity_data.sel(subject_id=subject_id).values
    else:
        # Simple numpy array
        return connectivity_data[subject_id - 1]  # Convert to 0-based indexing


def get_subject_harmonics(harmonics_data, subject_id: int) -> np.ndarray:
    """Get harmonics for a specific subject.
    
    Parameters
    ----------
    harmonics_data : xr.DataArray
        Harmonics data
    subject_id : int
        Subject ID (1-based)
        
    Returns
    -------
    np.ndarray
        Harmonics for the subject
    """
    if isinstance(harmonics_data, xr.DataArray):
        return harmonics_data.sel(subject_id=subject_id).values
    else:
        return harmonics_data[subject_id - 1]
