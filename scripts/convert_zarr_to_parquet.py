#!/usr/bin/env python3
"""
Convert zarr data to simple parquet format.

This script converts the overcomplicated zarr storage to simple parquet files
that are much easier to work with for datasets of this size.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ch.settings import load_settings


def convert_connectivity_zarr_to_parquet():
    """Convert connectivity zarr to parquet."""
    settings = load_settings()
    
    # Load zarr data
    zarr_path = settings.processed_dir / "connectivity_matrices.zarr"
    print(f"Loading zarr data from: {zarr_path}")
    
    if not zarr_path.exists():
        print(f"Zarr file not found: {zarr_path}")
        return
    
    ds = xr.open_zarr(zarr_path)
    print(f"Zarr data shape: {ds.connectivity.shape}")
    
    # Convert to long format for parquet
    print("Converting to long format...")
    
    # Get all the data (this will load it into memory - fine for your dataset size)
    connectivity_data = ds.connectivity.values
    
    # Create long format DataFrame
    rows = []
    n_subjects, n_regions_i, n_regions_j = connectivity_data.shape
    
    print(f"Processing {n_subjects} subjects with {n_regions_i}x{n_regions_j} matrices...")
    
    for subj_idx in range(n_subjects):
        if subj_idx % 50 == 0:
            print(f"  Processing subject {subj_idx + 1}/{n_subjects}")
        
        for i in range(n_regions_i):
            for j in range(n_regions_j):
                rows.append({
                    'subject_id': subj_idx + 1,  # 1-based indexing
                    'region_i': i,
                    'region_j': j,
                    'connectivity_value': connectivity_data[subj_idx, i, j]
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Save to parquet
    output_path = settings.connectivity_parquet
    print(f"Saving to: {output_path}")
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Conversion complete! File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    # Verify the conversion
    print("\nVerifying conversion...")
    test_df = pd.read_parquet(output_path)
    print(f"Loaded back: {test_df.shape}")
    print(f"Subject IDs: {test_df['subject_id'].min()} to {test_df['subject_id'].max()}")
    print(f"Region range: {test_df['region_i'].min()} to {test_df['region_i'].max()}")


def convert_harmonics_zarr_to_parquet():
    """Convert harmonics zarr to parquet."""
    settings = load_settings()
    
    # Load zarr data
    zarr_path = settings.processed_dir / "connectome_harmonics.zarr"
    print(f"Loading harmonics zarr data from: {zarr_path}")
    
    if not zarr_path.exists():
        print(f"Zarr file not found: {zarr_path}")
        return
    
    ds = xr.open_zarr(zarr_path)
    print(f"Zarr harmonics shape: {ds.harmonics.shape}")
    
    # Convert to long format for parquet
    print("Converting harmonics to long format...")
    
    # Get all the data
    harmonics_data = ds.harmonics.values
    
    # Create long format DataFrame
    rows = []
    n_subjects, n_harmonics, n_regions = harmonics_data.shape
    
    print(f"Processing {n_subjects} subjects with {n_harmonics} harmonics for {n_regions} regions...")
    
    for subj_idx in range(n_subjects):
        if subj_idx % 50 == 0:
            print(f"  Processing subject {subj_idx + 1}/{n_subjects}")
        
        for harm_idx in range(n_harmonics):
            for region_idx in range(n_regions):
                rows.append({
                    'subject_id': subj_idx + 1,  # 1-based indexing
                    'harmonic': harm_idx,
                    'region': region_idx,
                    'harmonic_value': harmonics_data[subj_idx, harm_idx, region_idx]
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Save to parquet
    output_path = settings.harmonics_parquet
    print(f"Saving to: {output_path}")
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Conversion complete! File size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    """Convert all zarr data to parquet."""
    print("🔄 Converting zarr data to simple parquet format...")
    print("=" * 60)
    
    # Convert connectivity
    print("\n1. Converting connectivity matrices...")
    convert_connectivity_zarr_to_parquet()
    
    # Convert harmonics
    print("\n2. Converting harmonics...")
    convert_harmonics_zarr_to_parquet()
    
    print("\n" + "=" * 60)
    print("✅ All conversions complete!")
    print("\nYou can now delete the zarr directories if you want:")
    print("  - data/processed/connectivity_matrices.zarr/")
    print("  - data/processed/connectome_harmonics.zarr/")


if __name__ == "__main__":
    main()
