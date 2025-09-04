"""
Template for data loading functions.

This template follows the project's best practices:
- Explicit path parameters
- Type hints
- No settings loading inside functions
- Pure, testable functions
"""

import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from explicit path.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the data file
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    return pd.read_parquet(data_path)


# Usage example:
# settings = load_settings()
# data_path = settings.data_root / "path" / "to" / "file.parquet"
# data = load_data(data_path)
