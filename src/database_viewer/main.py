"""
Main entry point for the bird species image viewer application.
"""
import sys
import os

# Add the src directory to the Python path so we can import bird_database
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database_reader.bird_database import BirdDatabase
from image_viewer import ImageViewer
from constants import DB_PATH, README_PATH, LABEL_NAME_PATH

def main():
    """
    Main function to run the bird species image viewer application.
    """
    # Load dataset and extract label names using BirdDatabase
    print("Loading dataset and extracting label names...")
    bird_db = BirdDatabase(DB_PATH, README_PATH, LABEL_NAME_PATH)
    print(f"Dataset loaded with {len(bird_db)} samples")
    print(f"Extracted {len(bird_db.label_names)} label names")

    # Create and run the viewer
    print("Starting image viewer...")
    viewer = ImageViewer(bird_db, start_idx=0)
    viewer.mainloop()

if __name__ == "__main__":
    main()
