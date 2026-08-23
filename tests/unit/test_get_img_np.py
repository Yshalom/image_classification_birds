"""Unit tests for BirdDatabase.get_img_np using the standard unittest framework."""
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import io
from unittest.mock import patch
from src.database_reader.bird_database import BirdDatabase


class TestGetImgNp(unittest.TestCase):
    """Test suite for the get_img_np method."""

    def setUp(self):
        """Set up a BirdDatabase instance with a mocked DataFrame."""
        # Create an instance without calling __init__ to avoid file I/O
        self.db = BirdDatabase.__new__(BirdDatabase)
        # Create a dummy image (RGB, 5*7, red) and get its bytes
        dummy_img = Image.new("RGB", (5, 7), color="red")
        buf = io.BytesIO()
        dummy_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        # Create a DataFrame with one row
        self.db._df = pd.DataFrame([{
            "label": 0,
            "image": {"bytes": image_bytes}
        }])
        # Set label names to avoid any issues in get_label (though not used in get_img_np)
        self.db._label_names = ("dummy_label", )

    def test_returns_numpy_array(self):
        """The function should return a NumPy ndarray."""
        result = self.db.get_img_np(0)
        self.assertIsInstance(result, np.ndarray)

    def test_correct_shape(self):
        """The returned array should have shape (height, width, 3)."""
        result = self.db.get_img_np(0)
        # height=7, width=5 from the dummy image => shape (7, 5, 3)
        self.assertEqual(result.shape, (7, 5, 3))

    def test_preserves_pixel_values(self):
        """Pixel values should be identical to the source PIL image."""
        result = self.db.get_img_np(0)
        # The dummy image is filled with red (255, 0, 0)
        expected_pixel = np.array([255, 0, 0], dtype=np.uint8)
        # Check a few pixels to ensure no mixing of dimensions
        self.assertTrue(np.array_equal(result[0, 0], expected_pixel))
        self.assertTrue(np.array_equal(result[1, 2], expected_pixel))
        self.assertTrue(np.array_equal(result[2, 4], expected_pixel))

    def test_different_dimensions(self):
        """Test that varying width/height works without mixing axes."""
        # Create a blue image (0, 0, 255) with a different size: width=2, height=4
        blue_img = Image.new("RGB", (2, 4), color="blue")
        buf = io.BytesIO()
        blue_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        # Update the DataFrame's image bytes for the same row (index 0)
        self.db._df.at[0, "image"] = {"bytes": image_bytes}
        result = self.db.get_img_np(0)
        # height=4, width=2 => shape (4, 2, 3)
        self.assertEqual(result.shape, (4, 2, 3))
        # Verify a pixel at (row=3, col=1) is blue
        self.assertTrue(np.array_equal(result[3, 1], np.array([0, 0, 255], dtype=np.uint8)))

    def test_invalid_row_raises_error(self):
        """An out-of-bounds index should raise IndexError through get_img."""
        with self.assertRaises(IndexError):
            self.db.get_img_np(1)  # Only one row in the DataFrame


if __name__ == "__main__":
    unittest.main()
