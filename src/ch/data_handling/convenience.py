"""
Convenience functions for data loading with settings.

These functions provide a settings-based interface for common data loading tasks.
Use the functions in data_utils.py for explicit path-based loading.
"""

from .data_utils import (
    load_metadata,
    load_connectivity, 
    load_harmonics,
    load_yeo_labels,
    get_subject_connectivity,
    get_subject_harmonics
)


def load_metadata_with_settings(settings=None):
    """Load metadata using settings (convenience function)."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    metadata_path = settings.data_root / "metadata" / "subject_metadata.parquet"
    return load_metadata(metadata_path)


def load_connectivity_with_settings(settings=None):
    """Load connectivity using settings (convenience function)."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    connectivity_path = settings.processed_dir / "connectivity_matrices.zarr"
    return load_connectivity(connectivity_path)


def load_harmonics_with_settings(settings=None):
    """Load harmonics using settings (convenience function)."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    harmonics_path = settings.processed_dir / "connectome_harmonics.zarr"
    return load_harmonics(harmonics_path)


def load_yeo_labels_with_settings(settings=None):
    """Load yeo labels using settings (convenience function)."""
    if settings is None:
        from ch.settings import load_settings
        settings = load_settings()
    
    yeo_path = settings.processed_dir / "yeo_labels.pkl"
    return load_yeo_labels(yeo_path)
