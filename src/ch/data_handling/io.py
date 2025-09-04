import scipy
import pandas as pd
import os
from datetime import datetime
import pickle
from scipy import io
from .settings import load_settings


def get_nets_and_ages():
    """Load networks and ages from the configured MATLAB file.

    Uses `CH_DATA_ROOT` and `CH_MAT_FILENAME` (or defaults) via settings.
    """
    s = load_settings()
    mat = scipy.io.loadmat(str(s.mat_path))
    return mat["nets"], mat["age"].flatten()


def save_dict_as_pickle(dictionary, filename):
    """
    Save a dictionary into a pickle file.

    Parameters:
    - dictionary (dict): The dictionary to be saved.
    - filename (str): The filename for the pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def load_dict_from_pickle(path, filename):
    """
    Load a dictionary from a pickle file.

    Parameters:
    - path (str): The directory path where the pickle file is located.
    - filename (str): The name of the pickle file.

    Returns:
    - dict: The dictionary loaded from the pickle file.
    """
    full_path = os.path.join(path, filename)
    with open(full_path, 'rb') as file:
        print(f"Loading {filename}")
        return pickle.load(file)


def add_column_from_dict(df, column_name, data_dict):
    """
    Adds a new column to a DataFrame using values from a dictionary.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to which the column will be added.
    - column_name (str): The name of the new column.
    - data_dict (dict): Dictionary containing data for the new column. The keys should match the DataFrame's index.
    
    Returns:
    - pd.DataFrame: DataFrame with the new column added.
    """

    # Backup the existing metadata_df into the configured backups folder
    s = load_settings()
    backup_dataframe(df, backup_dir=str(s.backups_dir))

    df[column_name] = df.index.map(data_dict)
    return df


def backup_dataframe(df, backup_dir='data/backups'):
    """
    Backs up a dataframe to a specified directory with a timestamp.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to back up.
    - backup_dir (str): Directory to save the backup. Default is 'backups' in the current working directory.
    
    Returns:
    - str: Path to the backup file.
    """
    # Check if backup directory exists, if not create it
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Format the current time as YYYYMMDD_HHMMSS for uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'metadata_backup_{timestamp}.csv')
    
    # Save the dataframe to a csv file
    df.to_csv(backup_path)
    
    return backup_path


def load_matlab_file(filename):
    data = scipy.io.loadmat(filename)
    print("Data keys:")
    print(list(data.keys()))
    
    return data
