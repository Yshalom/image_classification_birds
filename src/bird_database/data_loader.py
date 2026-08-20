"""
Data loader module for the bird species image viewer.
Handles loading the dataset from parquet file.
"""

import pandas as pd


def load_dataset(db_path: str) -> pd.DataFrame:
    """
    Load the bird species dataset from a parquet file.

    Args:
        db_path (str): Path to the parquet file containing the dataset

    Returns:
        pd.DataFrame: DataFrame containing the bird species data

    Raises:
        Exception: If there's an error loading the dataset
    """
    try:
        df = pd.read_parquet(db_path)
        return df
    except Exception as e:
        raise Exception(f"Failed to load dataset from {db_path}: {str(e)}") from e