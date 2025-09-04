# Data Organization Guide

This document describes the organized data structure for the connectome harmonics project.

## Overview

The raw MATLAB data has been organized into a structured format for efficient analysis:

- **594 subjects** with ages ranging from 18-88 years
- **376×376 connectivity matrices** (DTI scans) 
- **Yeo network labels** for brain regions
- **Template structure** for connectome harmonics computation

## Data Structure

### 1. Metadata (`data/metadata/subject_metadata.parquet`)

Subject information stored as a parquet file for fast loading:

```python
from ch.data_handling.data_utils import load_metadata

metadata = load_metadata()
print(metadata.head())
```

**Columns:**
- `subject_id`: Subject identifier (1-594)
- `age`: Subject age in years
- `dataset`: Dataset name ("camcan")
- `scan_type`: Type of scan ("DTI")
- `n_regions`: Number of brain regions (376)
- `has_yeo_labels`: Whether Yeo labels are available (True)

### 2. Connectivity Matrices (`data/processed/connectivity_matrices.zarr`)

DTI connectivity matrices stored as xarray dataset in zarr format:

```python
from ch.data_handling.data_utils import load_connectivity

connectivity = load_connectivity()
print(connectivity)
```

**Structure:**
- **Data variable**: `connectivity` (376, 376, 594)
- **Coordinates**: 
  - `region_i`: Brain region indices (0-375)
  - `region_j`: Brain region indices (0-375) 
  - `subject_id`: Subject identifiers (1-594)

**Usage:**
```python
# Get connectivity matrix for specific subject
subject_conn = connectivity.connectivity.sel(subject_id=1).values

# Get all matrices for multiple subjects
multi_subject_conn = connectivity.connectivity.sel(subject_id=[1, 2, 3])
```

### 3. Connectome Harmonics Template (`data/processed/connectome_harmonics.zarr`)

Template structure for storing computed harmonics:

```python
from ch.data_handling.data_utils import load_harmonics

harmonics = load_harmonics()
print(harmonics)
```

**Structure:**
- **Data variables**:
  - `harmonics`: Harmonic modes (100, 376, 594) - *currently zeros, to be filled*
  - `eigenvalues`: Eigenvalues (100, 594) - *currently zeros, to be filled*
- **Coordinates**:
  - `harmonic`: Harmonic indices (0-99)
  - `region`: Brain region indices (0-375)
  - `subject_id`: Subject identifiers (1-594)

### 4. Yeo Labels (`data/processed/yeo_labels.pkl`)

Brain network labels stored as pickle file:

```python
import pickle

with open('data/processed/yeo_labels.pkl', 'rb') as f:
    yeo_labels = pickle.load(f)
```

## Data Loading Functions

Convenient functions are available in `src/ch/data_handling/data_utils.py`:

```python
from ch.data_handling.data_utils import (
    load_metadata,
    load_connectivity, 
    load_harmonics,
    get_subject_connectivity,
    get_subject_harmonics
)

# Load all data
metadata = load_metadata()
connectivity = load_connectivity()
harmonics = load_harmonics()

# Get specific subject data
subject_id = 1
conn_matrix = get_subject_connectivity(connectivity, subject_id)
harm_data = get_subject_harmonics(harmonics, subject_id)
```

## File Sizes

- `subject_metadata.parquet`: ~8KB
- `connectivity_matrices.zarr/`: ~1.7GB (compressed)
- `connectome_harmonics.zarr/`: ~450MB (template)
- `yeo_labels.pkl`: ~2KB

## Next Steps

1. **Compute harmonics**: Fill the harmonics template with actual computed values
2. **Analysis**: Use the organized data for statistical analysis and visualization
3. **Extend**: Add new subjects or datasets using the same structure

## Data Processing Script

The data organization was performed by `scripts/organize_data.py`. This script:

1. Loads raw MATLAB data
2. Creates metadata DataFrame
3. Organizes connectivity matrices into xarray
4. Creates harmonics template structure
5. Saves everything in efficient formats (parquet, zarr)
6. Generates utility functions for data loading

To re-run the organization:
```bash
python scripts/organize_data.py
```

## Benefits of This Organization

1. **Efficient storage**: Zarr format enables fast, chunked access to large arrays
2. **Easy querying**: xarray provides intuitive indexing and selection
3. **Metadata integration**: Subject information easily linked to brain data
4. **Scalable**: Structure supports adding more subjects or datasets
5. **Analysis-ready**: Data is immediately usable for statistical analysis
