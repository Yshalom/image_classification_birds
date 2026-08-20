"""
Data loader module for the bird species image viewer.
Handles loading the dataset from parquet file.
"""

from bird_database.data_loader import load_dataset

# Re-export for backward compatibility
__all__ = ["load_dataset"]