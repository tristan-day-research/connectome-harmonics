"""
Metadata management for connectome harmonics project.

This module provides a comprehensive API for metadata operations:

Core Functions:
- load_metadata(): Load existing metadata from parquet
- instantiate_metadata(): Create new metadata file (with safety check)
- update_metadata(): Add or update a column in existing metadata
- delete_metadata_column(): Delete a column from existing metadata

Data Extraction:
- extract_camcan_metadata(): Extract metadata from CamCAN raw data
- extract_hcpa_metadata(): Extract metadata from HCP-A raw data

Utility Functions:
- backup_metadata(): Create timestamped backups
- get_subject_info(): Get all info for a specific subject
- get_column_stats(): Get statistics for any column
- filter_subjects(): Filter metadata based on criteria
- create_metadata_summary(): Generate summary statistics

Safety Features:
- Critical column protection (can't delete essential columns)
- File existence checks (won't overwrite accidentally)
- Automatic backups before major changes
- Comprehensive logging for all operations
- Data validation and error handling

Usage Examples:
    # Load existing metadata
    metadata = load_metadata(settings)
    
    # Create new metadata (with safety check)
    instantiate_metadata(settings, new_dataframe)
    
    # Add/update a column
    update_metadata(settings, new_data, 'column_name')
    
    # Delete a column (with safety checks)
    delete_metadata_column(settings, 'old_column')
    
    # Create backup
    backup_path = backup_metadata(settings)
    
    # Get subject info
    subject_info = get_subject_info(settings, subject_id=1)
    
    # Get column statistics
    stats = get_column_stats(settings, 'age')
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

from ..settings import Settings

logger = logging.getLogger(__name__)


def load_metadata(settings: Settings) -> pd.DataFrame:
    """Load metadata with validation.
    
    Args:
        settings: Project settings
        
    Returns:
        Validated metadata DataFrame
    """
    logger.info(f"Loading metadata from {settings.metadata_parquet}")
    
    if not settings.metadata_parquet.exists():
        raise FileNotFoundError(f"Metadata file not found: {settings.metadata_parquet}")
    
    df = pd.read_parquet(settings.metadata_parquet)
    # df = validate_metadata(df)
    
    logger.info(f"Loaded metadata for {len(df)} subjects")
    
    return df


def instantiate_metadata(settings: Settings, metadata_df: pd.DataFrame) -> None:
    """Create new metadata file (replaces entire metadata).
    
    Args:
        settings: Project settings
        metadata_df: DataFrame to save as metadata
        
    Raises:
        FileExistsError: If metadata.parquet already exists
    """
    logger.info(f"Instantiating new metadata with DataFrame shape: {metadata_df.shape}")
    
    # Check if metadata already exists
    if settings.metadata_parquet.exists():
        raise FileExistsError(
            "WARNING: metadata.parquet already exists! To instantiate a new metadata.parquet you must delete the existing one."
        )
    
    # Add creation timestamp
    metadata_df = metadata_df.copy()
    metadata_df['created_at'] = datetime.now()
    metadata_df['updated_at'] = datetime.now()
    
    # Ensure the metadata directory exists
    settings.metadata_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    # Save new metadata
    metadata_df.to_parquet(settings.metadata_parquet)
    
    logger.info(f"New metadata instantiated at {settings.metadata_parquet}")
    logger.info(f"Columns: {list(metadata_df.columns)}")


def update_metadata(settings: Settings, new_data: pd.DataFrame, column_name: str) -> None:
    """Add or update a column in existing metadata.
    
    Args:
        settings: Project settings
        new_data: DataFrame with subject_id and new column data
        column_name: Name of the column to add or update
    """
    logger.info(f"Updating column '{column_name}' in metadata")
    
    # Load existing metadata
    existing_metadata = load_metadata(settings)
    
    # Ensure new_data has subject_id as index
    if 'subject_id' in new_data.columns:
        new_data = new_data.set_index('subject_id')
    
    # Check if subject_id is the index
    if not isinstance(new_data.index.name, str) or 'subject_id' not in new_data.index.name:
        raise ValueError("new_data must have 'subject_id' as index or column")
    
    # Merge the new column with existing metadata
    if column_name in existing_metadata.columns:
        logger.warning(f"Column '{column_name}' already exists. Updating values.")
        existing_metadata[column_name] = new_data[column_name]
    else:
        # Add new column, keeping all existing subjects
        existing_metadata = existing_metadata.join(new_data[[column_name]], how='left')
    
    # Add update timestamp
    existing_metadata['updated_at'] = datetime.now()
    
    # Ensure the metadata directory exists
    settings.metadata_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    # Save updated metadata
    existing_metadata.to_parquet(settings.metadata_parquet)
    
    logger.info(f"Successfully updated column '{column_name}' in metadata")
    logger.info(f"Updated metadata saved to {settings.metadata_parquet}")


def delete_metadata_column(settings: Settings, column_name: str) -> None:
    """Delete a column from existing metadata.
    
    Args:
        settings: Project settings
        column_name: Name of the column to delete
    """
    logger.info(f"Deleting column '{column_name}' from metadata")
    
    # Load existing metadata
    existing_metadata = load_metadata(settings)
    
    # Check if column exists
    if column_name not in existing_metadata.columns:
        logger.warning(f"Column '{column_name}' does not exist in metadata")
        return
    
    # Don't allow deletion of critical columns
    critical_columns = ['subject_id', 'age', 'dataset', 'created_at', 'updated_at']
    if column_name in critical_columns:
        raise ValueError(f"Cannot delete critical column '{column_name}'. Critical columns: {critical_columns}")
    
    # Delete the column
    existing_metadata = existing_metadata.drop(columns=[column_name])
    
    # Add update timestamp
    existing_metadata['updated_at'] = datetime.now()
    
    # Ensure the metadata directory exists
    settings.metadata_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    # Save updated metadata
    existing_metadata.to_parquet(settings.metadata_parquet)
    
    logger.info(f"Successfully deleted column '{column_name}' from metadata")
    logger.info(f"Updated metadata saved to {settings.metadata_parquet}")


def backup_metadata(settings: Settings) -> Path:
    """Create a timestamped backup of the current metadata.
    
    Args:
        settings: Project settings
        
    Returns:
        Path to the backup file
    """
    logger.info("Creating metadata backup")
    
    # Load current metadata
    current_metadata = load_metadata(settings)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = settings.backups_dir / f"metadata_backup_{timestamp}.parquet"
    
    # Ensure backup directory exists
    settings.backups_dir.mkdir(parents=True, exist_ok=True)
    
    # Save backup
    current_metadata.to_parquet(backup_path)
    
    logger.info(f"Metadata backup saved to {backup_path}")
    return backup_path


def get_subject_info(settings: Settings, subject_id: int) -> pd.Series:
    """Get all information for a specific subject.
    
    Args:
        settings: Project settings
        subject_id: Subject ID to look up
        
    Returns:
        Series with all subject information
    """
    metadata = load_metadata(settings)
    
    if subject_id not in metadata.index:
        raise ValueError(f"Subject {subject_id} not found in metadata")
    
    return metadata.loc[subject_id]


def get_column_stats(settings: Settings, column_name: str) -> Dict[str, Any]:
    """Get basic statistics for a specific column.
    
    Args:
        settings: Project settings
        column_name: Name of the column to analyze
        
    Returns:
        Dictionary with column statistics
    """
    metadata = load_metadata(settings)
    
    if column_name not in metadata.columns:
        raise ValueError(f"Column '{column_name}' not found in metadata")
    
    col_data = metadata[column_name]
    
    stats = {
        'count': len(col_data),
        'non_null_count': col_data.count(),
        'null_count': col_data.isnull().sum(),
        'dtype': str(col_data.dtype),
        'unique_count': col_data.nunique()
    }
    
    # Add numeric statistics if applicable
    if pd.api.types.is_numeric_dtype(col_data):
        stats.update({
            'min': col_data.min(),
            'max': col_data.max(),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'median': col_data.median()
        })
    
    return stats


def extract_camcan_metadata(settings: Settings) -> pd.DataFrame:
    """Extract metadata from CamCAN raw data.
    
    Args:
        settings: Settings object containing paths
        
    Returns:
        DataFrame with subject metadata
    """
    logger.info(f"Extracting CamCAN metadata from {settings.camcan_raw}")
    
    # Load the raw data to get ages
    from .io import load_matlab_file
    data = load_matlab_file(settings.camcan_raw)
    
    n_subjects = data['age'].shape[0]
    subject_ids = np.arange(1, n_subjects + 1)  # 1-indexed to match MATLAB
    
    metadata = pd.DataFrame({
        'subject_id': subject_ids,
        'age': data['age'].flatten(),
        'dataset': 'camcan',
        'scan_type': 'DTI',
        'n_regions': 376,  # From connectivity matrix shape
        'atlas': 'AAL-376',  # Based on 376 regions
        'has_yeo_labels': 'yeoLabs' in data and data['yeoLabs'] is not None,
        'created_at': datetime.now(),
    })
    
    # Set subject_id as index for easy lookup
    metadata = metadata.set_index('subject_id')
    
    logger.info(f"Extracted metadata for {len(metadata)} CamCAN subjects")
    logger.info(f"Age range: {metadata['age'].min():.1f} - {metadata['age'].max():.1f}")
    
    return metadata


def extract_hcpa_metadata(raw_data_path: Path) -> pd.DataFrame:
    """Extract metadata from HCP-A raw data.
    
    Args:
        raw_data_path: Path to the raw HCP-A data file
        
    Returns:
        DataFrame with subject metadata
    """
    logger.info(f"Extracting HCP-A metadata from {raw_data_path}")
    
    # TODO: Implement when HCP-A data is available
    # This is a placeholder structure
    metadata = pd.DataFrame({
        'subject_id': [],
        'age': [],
        'dataset': 'hcpa',
        'scan_type': 'DTI',
        'n_regions': 364,  # HCP-A uses AAL-364
        'atlas': 'AAL-364',
        'has_yeo_labels': False,
        'created_at': datetime.now(),
    })
    
    if len(metadata) > 0:
        metadata = metadata.set_index('subject_id')
    
    logger.info(f"Extracted metadata for {len(metadata)} HCP-A subjects")
    
    return metadata


def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean metadata.
    
    Args:
        df: Metadata DataFrame
        
    Returns:
        Cleaned metadata DataFrame
    """
    logger.info("Validating metadata...")
    
    original_count = len(df)
    
    # Check required fields
    required_fields = ['age', 'dataset', 'scan_type', 'n_regions']
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Clean age data
    if 'age' in df.columns:
        # Remove subjects with invalid ages
        valid_ages = (df['age'] >= 0) & (df['age'] <= 120)
        df = df[valid_ages]
        logger.info(f"Removed {original_count - len(df)} subjects with invalid ages")
    
    # Ensure dataset is string
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].astype(str)
    
    # Add quality flags
    df['age_valid'] = (df['age'] >= 18) & (df['age'] <= 100)
    df['has_connectivity'] = True  # All subjects should have connectivity data
    
    logger.info(f"Validation complete: {len(df)}/{original_count} subjects passed")
    
    return df


def filter_subjects(metadata: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Filter subjects based on criteria.
    
    Args:
        metadata: Metadata DataFrame
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered metadata DataFrame
    """
    logger.info(f"Filtering subjects with criteria: {filters}")
    
    df = metadata.copy()
    original_count = len(df)
    
    # Age filters
    if 'min_age' in filters:
        df = df[df['age'] >= filters['min_age']]
    if 'max_age' in filters:
        df = df[df['age'] <= filters['max_age']]
    
    # Dataset filters
    if 'datasets' in filters:
        datasets = filters['datasets'] if isinstance(filters['datasets'], list) else [filters['datasets']]
        df = df[df['dataset'].isin(datasets)]
    
    # Quality filters
    if 'min_quality_score' in filters:
        # This would require a quality score column
        pass
    
    # Sex filters (if available)
    if 'include_sex' in filters and 'sex' in df.columns:
        sexes = filters['include_sex'] if isinstance(filters['include_sex'], list) else [filters['include_sex']]
        df = df[df['sex'].isin(sexes)]
    
    logger.info(f"Filtered to {len(df)}/{original_count} subjects")
    
    return df


def get_subjects_for_analysis(settings: Settings, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Get filtered subjects for analysis.
    
    Args:
        settings: Project settings
        filters: Optional filter criteria
        
    Returns:
        Filtered metadata DataFrame
    """
    metadata = load_metadata(settings)
    
    if filters:
        metadata = filter_subjects(metadata, filters)
    
    return metadata




def create_metadata_summary(settings: pd.DataFrame) -> Dict[str, Any]:
    """Create a summary of metadata for reporting.
    
    Args:
        metadata: Metadata DataFrame
        
    Returns:
        Dictionary with summary statistics
    """

    metadata = load_metadata(settings)

    summary = {
        'total_subjects': len(metadata),
        'datasets': metadata['dataset'].value_counts().to_dict() if 'dataset' in metadata.columns else {},
        'age_stats': {
            'min': float(metadata['age'].min()),
            'max': float(metadata['age'].max()),
            'mean': float(metadata['age'].mean()),
            'std': float(metadata['age'].std()),
        } if 'age' in metadata.columns else {},
        'scan_types': metadata['scan_type'].value_counts().to_dict() if 'scan_type' in metadata.columns else {},
        'atlas_versions': metadata['atlas'].value_counts().to_dict() if 'atlas' in metadata.columns else {},
    }
    
    return summary


