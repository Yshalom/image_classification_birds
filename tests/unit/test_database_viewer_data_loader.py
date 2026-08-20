import unittest
import sys
import os

src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test that we can import from database_viewer.data_loader and that it works
from database_viewer.data_loader import load_dataset

class TestDatabaseViewerDataLoader(unittest.TestCase):
    """Test cases for the database_viewer.data_loader module (which re-exports bird_database.data_loader.load_dataset)."""

    def test_load_dataset_is_callable(self):
        """Test that load_dataset is callable and is the same function as from bird_database."""
        from bird_database.data_loader import load_dataset as bird_load_dataset
        self.assertIs(load_dataset, bird_load_dataset)

    def test_load_dataset_success(self):
        """Test that load_dataset successfully loads a parquet file via the re-export."""
        db_path = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
        try:
            df = load_dataset(db_path)
            self.assertIsInstance(df, __import__('pandas').DataFrame)
            self.assertGreater(len(df), 0)
        except FileNotFoundError:
            self.skipTest("Test data file not found")

if __name__ == '__main__':
    unittest.main()