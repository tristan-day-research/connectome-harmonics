#!/usr/bin/env python3
"""
Convert raw MATLAB data to simple parquet format.
One-time script to get your data ready.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ch.settings import load_settings
from ch.data_handling.io import load_matlab_file, get_nets_and_ages


def main():
    """Convert raw data to simple parquet format."""
    print("🔄 Processing raw data to simple parquet format...")
    
    settings = load_settings()
    settings.ensure_dirs()
    
    # Load raw data
    print("Loading raw MATLAB data...")
    nets, ages = get_nets_and_ages(settings.camcan_raw)
    print(f"Loaded: {nets.shape} connectivity matrices, {ages.shape} ages")
    
    # Create metadata DataFrame
    print("Creating metadata...")
    metadata = pd.DataFrame({
        'subject_id': range(1, len(ages) + 1),
        'age': ages.flatten()
    })
    metadata.set_index('subject_id', inplace=True)
    
    # Save metadata
    metadata_path = settings.metadata_parquet
    metadata.to_parquet(metadata_path)
    print(f"✅ Saved metadata: {metadata_path}")
    
    # Convert connectivity to long format for parquet
    print("Converting connectivity matrices to parquet...")
    rows = []
    n_subjects, n_regions_i, n_regions_j = nets.shape
    
    for subj_idx in range(n_subjects):
        if subj_idx % 50 == 0:
            print(f"  Processing subject {subj_idx + 1}/{n_subjects}")
        
        for i in range(n_regions_i):
            for j in range(n_regions_j):
                rows.append({
                    'subject_id': subj_idx + 1,
                    'region_i': i,
                    'region_j': j,
                    'connectivity_value': nets[i, j, subj_idx]
                })
    
    # Save connectivity
    connectivity_df = pd.DataFrame(rows)
    connectivity_path = settings.connectivity_parquet
    connectivity_df.to_parquet(connectivity_path, index=False)
    print(f"✅ Saved connectivity: {connectivity_path}")
    print(f"   File size: {connectivity_path.stat().st_size / 1024**2:.1f} MB")
    
    print("\n✅ DONE! Your data is ready:")
    print(f"   Metadata: {metadata_path}")
    print(f"   Connectivity: {connectivity_path}")


if __name__ == "__main__":
    main()
