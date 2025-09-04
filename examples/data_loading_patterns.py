#!/usr/bin/env python3
"""
Examples of different data loading patterns and best practices.

This demonstrates the progression from problematic to good practices.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ch.settings import load_settings
from ch.data_handling.data_utils import load_metadata, load_connectivity, load_harmonics
from ch.data_handling import load_metadata_with_settings, load_connectivity_with_settings


def example_1_bad_practice():
    """❌ BAD: Settings loaded inside helper functions."""
    print("❌ BAD PRACTICE: Settings loaded inside helper functions")
    print("Problems:")
    print("- Hidden dependencies")
    print("- Hard to test")
    print("- Tight coupling")
    print("- Less flexible")
    print()


def example_2_good_practice_explicit_paths():
    """✅ GOOD: Explicit paths passed to functions."""
    print("✅ GOOD PRACTICE: Explicit paths")
    
    # Load settings at the application level
    settings = load_settings()
    
    # Pass explicit paths to functions
    metadata_path = settings.data_root / "metadata" / "subject_metadata.parquet"
    connectivity_path = settings.processed_dir / "connectivity_matrices.zarr"
    
    # Functions are pure and testable
    metadata = load_metadata(metadata_path)
    connectivity = load_connectivity(connectivity_path)
    
    print(f"Loaded metadata: {metadata.shape}")
    print(f"Loaded connectivity: {connectivity.dims}")
    print()


def example_3_good_practice_convenience_functions():
    """✅ GOOD: Convenience functions for common use cases."""
    print("✅ GOOD PRACTICE: Convenience functions")
    
    # Use convenience functions when you want settings-based loading
    metadata = load_metadata_with_settings()
    connectivity = load_connectivity_with_settings()
    
    print(f"Loaded metadata: {metadata.shape}")
    print(f"Loaded connectivity: {connectivity.dims}")
    print()


def example_4_good_practice_custom_paths():
    """✅ GOOD: Easy to use custom paths."""
    print("✅ GOOD PRACTICE: Custom paths")
    
    # Easy to use different paths for testing or different datasets
    custom_metadata_path = "data/metadata/subject_metadata.parquet"
    custom_connectivity_path = "data/processed/connectivity_matrices.zarr"
    
    metadata = load_metadata(custom_metadata_path)
    connectivity = load_connectivity(custom_connectivity_path)
    
    print(f"Loaded from custom paths: {metadata.shape}, {connectivity.dims}")
    print()


def example_5_good_practice_testing():
    """✅ GOOD: Easy to test with mock data."""
    print("✅ GOOD PRACTICE: Easy testing")
    
    # Easy to test with mock data
    import tempfile
    import pandas as pd
    
    # Create test data
    test_data = pd.DataFrame({
        'subject_id': [1, 2, 3],
        'age': [25, 30, 35]
    }).set_index('subject_id')
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        test_data.to_parquet(f.name)
        
        # Test the function with mock data
        loaded_data = load_metadata(f.name)
        print(f"Test data loaded: {loaded_data.shape}")
        print("✅ Function works with any path!")
        
        # Clean up
        Path(f.name).unlink()
    print()


def main():
    """Run all examples."""
    print("Data Loading Patterns and Best Practices")
    print("=" * 50)
    print()
    
    example_1_bad_practice()
    example_2_good_practice_explicit_paths()
    example_3_good_practice_convenience_functions()
    example_4_good_practice_custom_paths()
    example_5_good_practice_testing()
    
    print("Summary of Best Practices:")
    print("1. ✅ Pass explicit paths to low-level functions")
    print("2. ✅ Load settings at application level, not in helpers")
    print("3. ✅ Provide convenience functions for common use cases")
    print("4. ✅ Make functions pure and testable")
    print("5. ✅ Use type hints and clear documentation")


if __name__ == "__main__":
    main()
