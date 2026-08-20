"""
Main entry point for the bird species image viewer application.
"""
import sys
import os

# Add the src directory to the Python path so we can import bird_database
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_loader import load_dataset
from label_extractor import extract_label_names_from_readme
from image_viewer import ImageViewer

def main():
    """
    Main function to run the bird species image viewer application.
    """
    # Define paths
    db_path = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
    readme_path = "database/birds-525-species-image-classification/README.md"
    label_name_path = [
        "dataset_info:",
        "features:",
        "name: label",
        "dtype:",
        "class_label:",
        "names:"
    ]

    # Load dataset
    print("Loading dataset...")
    df = load_dataset(db_path)
    print(f"Dataset loaded with {len(df)} samples")

    # Extract label names
    print("Extracting label names from README...")
    label_names = extract_label_names_from_readme(readme_path, label_name_path)
    print(f"Extracted {len(label_names)} label names")

    # Create and run the viewer
    print("Starting image viewer...")
    # All row indices in the parquet file
    all_indices = list(range(len(df)))
    viewer = ImageViewer(df, all_indices, label_names, start_idx=0)
    viewer.mainloop()


if __name__ == "__main__":
    main()
