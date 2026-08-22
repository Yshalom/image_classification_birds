"""
Main entry point for the bird species image viewer application.
"""
import sys
import os
import pandas as pd

DB_PATH = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
README_PATH = "database/birds-525-species-image-classification/README.md"

# Path of the class label names in the README.md nested structure
#   +------------------------+
#   | dataset_info:          |
#   |   features:            |
#   |     name: label        |
#   |       dtype:           |
#   |         class_label:   |
#   |           names:       |
#   |             '0': $NAME |
#   |             '1': $NAME |
#   |             ...        |
#   +------------------------+
LABEL_NAME_PATH = (
    "dataset_info:",
    "features:",
    "name: label",
    "dtype:",
    "class_label:",
    "names:"
)

# Add the src directory to the Python path so we can import bird_database
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bird_database.label_extractor import extract_label_names_from_readme
from image_viewer import ImageViewer

def main():
    """
    Main function to run the bird species image viewer application.
    """

    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet(DB_PATH)
    print(f"Dataset loaded with {len(df)} samples")

    # Extract label names
    print("Extracting label names from README...")
    label_names = extract_label_names_from_readme(README_PATH, LABEL_NAME_PATH)
    print(f"Extracted {len(label_names)} label names")

    # Create and run the viewer
    print("Starting image viewer...")
    viewer = ImageViewer(df, label_names, start_idx=0)
    viewer.mainloop()

if __name__ == "__main__":
    main()
