import unittest
import sys
import os

src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from bird_database.data_loader import load_dataset

class TestBirdDatabaseDataLoader(unittest.TestCase):
    """Test cases for the bird_database.data_loader module."""

    def test_load_dataset_success(self):
        """Test that load_dataset successfully loads a parquet file."""
        # We'll use the actual path from the project for this test
        db_path = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
        try:
            df = load_dataset(db_path)
            self.assertIsInstance(df, __import__('pandas').DataFrame)
            self.assertGreater(len(df), 0)
        except FileNotFoundError:
            self.skipTest("Test data file not found")

    def test_load_dataset_invalid_path(self):
        """Test that load_dataset raises an exception for an invalid path."""
        with self.assertRaises(Exception):
            load_dataset("non/existent/path.parquet")

if __name__ == '__main__':
    unittest.main()