import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd
from PIL import Image

# Add the src directory to the path so we can import BirdDatabase
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from database_reader.bird_database import BirdDatabase

DUMMY_README_CONTENT = """---
database_header:
  section:
  - name: dummy name here
    dtype: dummy_type_here
  - name: label
    dtype:
      class_label:
        names:
          '0': SPECIES_A
          '1': SPECIES_B
          '2': SPECIES_C
---"""

LABEL_NAME_PATH = (
    "database_header:",
    "section:",
    "name: label",
    "dtype:",
    "class_label:",
    "names:"
)

# 1 black pixel, RGB mode, PNG format
DUMMY_IMAGE_BYTES = bytes(( \
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, \
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, \
    0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x60, 0x60, 0x60, 0xF8, \
    0x0F, 0x00, 0x01, 0x04, 0x01, 0x00, 0x5F, 0xE5, 0xC3, 0x4B, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, \
    0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82 \
))

class TestBirdDatabase(unittest.TestCase):
    """Test cases for the BirdDatabase class."""

    def setUp(self):
        """Set up temporary files for testing."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create a dummy parquet file
        self.parquet_path = os.path.join(self.test_dir, "test.parquet")

        # Create a simple DataFrame with image bytes and labels
        self.df = pd.DataFrame({
            "image": [{"bytes": DUMMY_IMAGE_BYTES}, {"bytes": DUMMY_IMAGE_BYTES}, {"bytes": DUMMY_IMAGE_BYTES}],
            "label": [0, 1, 2]
        })
        self.df.to_parquet(self.parquet_path)

        # Create a dummy README.md file with the expected structure
        self.readme_path = os.path.join(self.test_dir, "README.md")
        with open(self.readme_path, 'w') as f:
            f.write(DUMMY_README_CONTENT)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that BirdDatabase can be initialized and loads data correctly."""
        bird_db = BirdDatabase(self.parquet_path, self.readme_path, LABEL_NAME_PATH)

        # Check that the DataFrame was loaded
        self.assertIsInstance(bird_db.df, pd.DataFrame)
        self.assertEqual(len(bird_db.df), 3)

        # Check that label names were extracted (should be stripped of whitespace)
        self.assertEqual(bird_db.label_names, ("SPECIES_A", "SPECIES_B", "SPECIES_C"))

    def test_get_id(self):
        """Test the get_id method."""
        bird_db = BirdDatabase(self.parquet_path, self.readme_path, LABEL_NAME_PATH)

        self.assertEqual(bird_db.get_id(0), 0)
        self.assertEqual(bird_db.get_id(1), 1)
        self.assertEqual(bird_db.get_id(2), 2)
        with self.assertRaises(IndexError):
            bird_db.get_id(3)

    def test_get_label(self):
        """Test the get_label method."""
        bird_db = BirdDatabase(self.parquet_path, self.readme_path, LABEL_NAME_PATH)

        # Test known label IDs
        self.assertEqual(bird_db.get_label(0), "SPECIES_A")
        self.assertEqual(bird_db.get_label(1), "SPECIES_B")
        self.assertEqual(bird_db.get_label(2), "SPECIES_C")

        # Test out of range label ID
        self.assertEqual(bird_db.get_label(3), "UNKNOWN CLASS 3")
        self.assertEqual(bird_db.get_label(-1), "UNKNOWN CLASS -1")

    def test_get_img(self):
        """Test the get_img method returns a PIL Image."""
        bird_db = BirdDatabase(self.parquet_path, self.readme_path, LABEL_NAME_PATH)

        # Test that we get a PIL Image object
        img = bird_db.get_img(0)
        self.assertIsInstance(img, Image.Image)

if __name__ == '__main__':
    unittest.main()
