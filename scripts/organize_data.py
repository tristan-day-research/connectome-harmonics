#!/usr/bin/env python3
"""
Data organization script for connectome harmonics project.

This script processes the raw MATLAB data and organizes it into:
1. Metadata parquet file with subject information
2. Connectivity matrices as xarray dataset
3. Structure for connectome harmonics storage

Usage:
    python scripts/organize_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import scipy.io
from tqdm import tqdm

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ch.settings import load_settings
from ch.data_handling.io import backup_dataframe


def load_raw_data(mat_path: Path) -> Dict[str, Any]:
    """Load raw MATLAB data and return organized dictionary."""
    print(f"Loading data from {mat_path}")
    data = scipy.io.loadmat(str(mat_path))
    
    # Filter out MATLAB metadata
    clean_data = {k: v for k, v in data.items() 
                  if not k.startswith('__') and k != 'None'}
    
    print("Available data keys:", list(clean_data.keys()))
    for key, value in clean_data.items():
        if hasattr(value, 'shape'):
            print(f"{key} shape: {value.shape}")
        else:
            print(f"{key} type: {type(value)}")
    
    return clean_data


def create_metadata_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Create metadata DataFrame with subject information."""
    print("Creating metadata DataFrame...")
    
    n_subjects = data['age'].shape[0]
    
    # Create subject IDs (1-indexed to match MATLAB convention)
    subject_ids = np.arange(1, n_subjects + 1)
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'subject_id': subject_ids,
        'age': data['age'].flatten(),
        'dataset': 'camcan',  # Based on filename
        'scan_type': 'DTI',
        'n_regions': 376,  # From connectivity matrix shape
    })
    
    # Note: yeo labels will be stored separately due to complex structure
    # For now, just add a placeholder
    metadata['has_yeo_labels'] = 'yeoLabs' in data and data['yeoLabs'] is not None
    
    # Set subject_id as index for easy lookup
    metadata = metadata.set_index('subject_id')
    
    print(f"Created metadata for {len(metadata)} subjects")
    print(f"Age range: {metadata['age'].min():.1f} - {metadata['age'].max():.1f}")
    
    return metadata


def create_connectivity_xarray(data: Dict[str, Any], metadata: pd.DataFrame) -> xr.Dataset:
    """Create xarray Dataset for connectivity matrices."""
    print("Creating connectivity xarray dataset...")
    
    nets = data['nets']  # Shape: (376, 376, 594)
    n_regions, _, n_subjects = nets.shape
    
    # Create coordinate arrays
    region_coords = np.arange(n_regions)
    subject_coords = metadata.index.values
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'connectivity': (['region_i', 'region_j', 'subject_id'], nets),
        },
        coords={
            'region_i': region_coords,
            'region_j': region_coords, 
            'subject_id': subject_coords,
        },
        attrs={
            'description': 'DTI connectivity matrices',
            'n_regions': n_regions,
            'n_subjects': n_subjects,
            'data_source': 'raw_data_nhw2022-network-harmonics-data.mat',
            'yeo_labels_available': 'yeoLabs' in data and data['yeoLabs'] is not None,
        }
    )
    
    print(f"Created connectivity dataset: {ds.connectivity.shape}")
    return ds


def create_harmonics_xarray_template(connectivity_ds: xr.Dataset, 
                                   n_harmonics: int = 100) -> xr.Dataset:
    """Create template xarray Dataset for connectome harmonics."""
    print(f"Creating harmonics xarray template with {n_harmonics} harmonics...")
    
    n_subjects = connectivity_ds.sizes['subject_id']
    n_regions = connectivity_ds.sizes['region_i']
    
    # Create template with zeros (will be filled during computation)
    harmonics_data = np.zeros((n_harmonics, n_regions, n_subjects))
    
    ds = xr.Dataset(
        {
            'harmonics': (['harmonic', 'region', 'subject_id'], harmonics_data),
            'eigenvalues': (['harmonic', 'subject_id'], np.zeros((n_harmonics, n_subjects))),
        },
        coords={
            'harmonic': np.arange(n_harmonics),
            'region': connectivity_ds.region_i.values,
            'subject_id': connectivity_ds.subject_id.values,
        },
        attrs={
            'description': 'Connectome harmonics and eigenvalues',
            'n_harmonics': n_harmonics,
            'n_regions': n_regions,
            'n_subjects': n_subjects,
            'computation_status': 'template_created',
        }
    )
    
    print(f"Created harmonics template: {ds.harmonics.shape}")
    return ds


def save_yeo_labels(data: Dict[str, Any], settings: Settings) -> None:
    """Save yeo labels separately as they have complex structure."""
    if 'yeoLabs' in data and data['yeoLabs'] is not None:
        yeo_path = settings.processed_dir / "yeo_labels.pkl"
        print(f"Saving yeo labels to {yeo_path}")
        import pickle
        with open(yeo_path, 'wb') as f:
            pickle.dump(data['yeoLabs'], f)


def save_data_organized(settings: Settings, metadata: pd.DataFrame, 
                       connectivity_ds: xr.Dataset, 
                       harmonics_ds: xr.Dataset,
                       raw_data: Dict[str, Any]) -> None:
    """Save all organized data to appropriate locations."""
    print("Saving organized data...")
    
    # Ensure directories exist
    settings.ensure_dirs()
    metadata_dir = settings.data_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    # Save metadata as parquet
    print(f"Saving metadata to {settings.metadata_parquet}")
    metadata.to_parquet(settings.metadata_parquet)
    
    # Backup metadata
    backup_dataframe(metadata, backup_dir=settings.backups_dir)
    
    # Save connectivity matrices as zarr (efficient for large arrays)
    connectivity_path = settings.processed_dir / "connectivity_matrices.zarr"
    print(f"Saving connectivity data to {connectivity_path}")
    connectivity_ds.to_zarr(connectivity_path, mode='w')
    
    # Save harmonics template as zarr
    harmonics_path = settings.processed_dir / "connectome_harmonics.zarr"
    print(f"Saving harmonics template to {harmonics_path}")
    harmonics_ds.to_zarr(harmonics_path, mode='w')
    
    # Save yeo labels separately
    save_yeo_labels(raw_data, settings)
    
    print("Data organization complete!")
    print(f"Metadata: {settings.metadata_parquet}")
    print(f"Connectivity: {connectivity_path}")
    print(f"Harmonics: {harmonics_path}")
    print(f"Yeo labels: {settings.processed_dir / 'yeo_labels.pkl'}")


def create_data_loading_functions():
    """Create utility functions for loading organized data."""
    functions_code = '''
def load_metadata(settings=None):
    """Load subject metadata from parquet file."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    metadata_path = settings.data_root / "metadata" / "subject_metadata.parquet"
    return pd.read_parquet(metadata_path)


def load_connectivity(settings=None):
    """Load connectivity matrices from zarr file."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    connectivity_path = settings.processed_dir / "connectivity_matrices.zarr"
    return xr.open_zarr(connectivity_path)


def load_harmonics(settings=None):
    """Load connectome harmonics from zarr file."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    harmonics_path = settings.processed_dir / "connectome_harmonics.zarr"
    return xr.open_zarr(harmonics_path)


def get_subject_connectivity(connectivity_ds, subject_id):
    """Get connectivity matrix for a specific subject."""
    return connectivity_ds.connectivity.sel(subject_id=subject_id).values


def get_subject_harmonics(harmonics_ds, subject_id):
    """Get harmonics for a specific subject."""
    return harmonics_ds.harmonics.sel(subject_id=subject_id).values
'''
    
    # Save utility functions
    utils_path = project_root / "src" / "ch" / "data_handling" / "data_utils.py"
    with open(utils_path, 'w') as f:
        f.write(functions_code)
    
    print(f"Created data loading utilities at {utils_path}")


def main():
    """Main function to organize all data."""
    print("Starting data organization...")
    
    # Load settings
    settings = load_settings()
    print(f"Using settings: {settings}")
    
    # Load raw data
    raw_data = load_raw_data(settings.camcan_raw)
    
    # Create organized data structures
    metadata = create_metadata_dataframe(raw_data)
    connectivity_ds = create_connectivity_xarray(raw_data, metadata)
    harmonics_ds = create_harmonics_xarray_template(connectivity_ds)
    
    # Save everything
    save_data_organized(settings, metadata, connectivity_ds, harmonics_ds, raw_data)
    
    # Create utility functions
    create_data_loading_functions()
    
    print("\nData organization summary:")
    print(f"✓ {len(metadata)} subjects processed")
    print(f"✓ {connectivity_ds.sizes['region_i']}x{connectivity_ds.sizes['region_j']} connectivity matrices")
    print(f"✓ Template for {harmonics_ds.sizes['harmonic']} harmonics per subject")
    print("✓ All data saved in organized format")


if __name__ == "__main__":
    main()
