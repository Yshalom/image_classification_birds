"""
Bird database module for handling the pandas DataFrame and label extraction.
"""

import pandas as pd
from PIL import Image
import io
from typing import Tuple

from database_reader.label_extractor import extract_label_names_from_readme

class BirdDatabase:
    """
    A class to handle the bird species dataset.
    This class is responsible for reading the parquet file and extracting label names.
    It provides methods to get the class ID, label string, and image for a given row index.
    """

    def __init__(self, db_path: str, readme_path: str, label_name_path: Tuple[str]):
        """
        Initialize the BirdDatabase by reading the parquet file and extracting label names.

        Args:
            db_path: Path to the parquet file containing the dataset.
            readme_path: Path to the README.md file containing label information.
            label_name_path: Tuple of strings representing the nested path to the label names in the README.
        """
        # Read the pandas database
        self.df = pd.read_parquet(db_path)

        # Extract the label names from the README and strip whitespace
        self.label_names = extract_label_names_from_readme(readme_path, label_name_path)

    def get_id(self, row_idx: int) -> int:
        """
        Return the class ID of the row at the given index.

        Args:
            row_idx: Index of the row in the DataFrame.

        Returns:
            The class ID (integer) of the row.
        """
        if not isinstance(row_idx, int) or row_idx < 0 or len(self.df) <= row_idx:
            raise IndexError(f"The row-index='{row_idx}' is invalid")
        row = self.df.iloc[row_idx]
        return int(row["label"])
        
    def get_label(self, row_idx: int) -> str:
        """
        Return the literal string class ID's label of the row at the given index.
        If the class ID is out of bounds of the label names list, returns "UNKNOWN CLASS $id".

        Args:
            row_idx: Index of the row in the DataFrame.

        Returns:
            The string label corresponding to the class ID of the row.
        """
        # Get the class ID: use actual if in bounds, otherwise use row_idx as fallback
        if 0 <= row_idx < len(self.df):
            row = self.df.iloc[row_idx]
            label_id = int(row["label"])
        else:
            return f"UNKNOWN CLASS {row_idx}"

        # Return the label name if in bounds, otherwise unknown class format
        if 0 <= label_id < len(self.label_names):
            return self.label_names[label_id]
        else:
            return f"UNKNOWN CLASS {label_id}"

    def get_img(self, row_idx: int) -> Image.Image:
        """
        Return the image object of the row at the given index.

        Args:
            row_idx: Index of the row in the DataFrame.

        Returns:
            A PIL.Image object of the image in the row.

        Raises:
            IndexError: If row_idx is out of bounds.
        """
        row = self.df.iloc[row_idx]  # This will raise IndexError if out of bounds
        image_bytes = row["image"]["bytes"]
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pil_img

    def __len__(self) -> int:
        return len(self.df)
