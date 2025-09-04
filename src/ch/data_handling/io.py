from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple

import pickle

import pandas as pd
import scipy


def get_nets_and_ages(mat_path: Path | str) -> Tuple[Any, Any]:
    """Load networks and ages from a MATLAB file at `mat_path`."""
    mat = scipy.io.loadmat(str(mat_path))
    return mat["nets"], mat["age"].flatten()


def save_dict_as_pickle(dictionary: Dict[str, Any], filename: Path | str) -> None:
    """
    Save a dictionary into a pickle file.

    Parameters:
    - dictionary (dict): The dictionary to be saved.
    - filename (str): The filename for the pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dict_from_pickle(path: Path | str, filename: str) -> Dict[str, Any]:
    """
    Load a dictionary from a pickle file.

    Parameters:
    - path (str): The directory path where the pickle file is located.
    - filename (str): The name of the pickle file.

    Returns:
    - dict: The dictionary loaded from the pickle file.
    """
    full_path = Path(path) / filename
    with open(full_path, 'rb') as file:
        print(f"Loading {filename}")
        return pickle.load(file)


def add_column_from_dict(
    df: pd.DataFrame,
    column_name: str,
    data_dict: Dict[Any, Any],
    backups_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Adds a new column to a DataFrame using values from a dictionary.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to which the column will be added.
    - column_name (str): The name of the new column.
    - data_dict (dict): Dictionary containing data for the new column. The keys should match the DataFrame's index.
    
    Returns:
    - pd.DataFrame: DataFrame with the new column added.
    """

    # Optional backup of the existing dataframe
    if backups_dir is not None:
        backup_dataframe(df, backup_dir=backups_dir)

    df[column_name] = df.index.map(data_dict)
    return df


def backup_dataframe(
    df: pd.DataFrame,
    backup_dir: Path | str | None = None,
) -> Path:
    """
    Backs up a dataframe to a specified directory with a timestamp.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to back up.
    - backup_dir (str): Directory to save the backup. Default is 'backups' in the current working directory.
    
    Returns:
    - str: Path to the backup file.
    """
    # Resolve backup directory: default to settings.backups_dir (repo-root based)
    if backup_dir is None:
        try:
            from ch.settings import load_settings
            backup_dir = load_settings().backups_dir
        except Exception:
            # Fallback to a repo-root-like relative path if settings import fails
            backup_dir = Path("data/backups")
    else:
        backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Format the current time as YYYYMMDD_HHMMSS for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'metadata_backup_{timestamp}.csv'
    
    # Save the dataframe to a csv file
    df.to_csv(backup_path)
    
    return backup_path


def load_matlab_file(filename: Path | str) -> Dict[str, Any]:
    filename = str(filename)
    data = scipy.io.loadmat(filename)
    print("Data keys:")
    print(list(data.keys()))
    
    return data
