import unittest
import sys
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from database_viewer.image_viewer import ImageViewer
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

class TestImageViewer(unittest.TestCase):
    """Test cases for the database_viewer.image_viewer.ImageViewer class."""

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

        self.bird_db = BirdDatabase(self.parquet_path, self.readme_path, LABEL_NAME_PATH)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_initialization(self, MockFrame, MockButton, MockLabel):
        """Test that the ImageViewer can be initialized (with mocked objects)."""
        # The patch is already applied from above

        # Configure the mocks to return mock objects
        mock_label_instance = MockLabel.return_value
        mock_label_instance.pack = MagicMock()
        mock_label_instance.configure = MagicMock()

        mock_button_instance = MockButton.return_value
        mock_button_instance.grid = MagicMock()

        mock_frame_instance = MockFrame.return_value
        mock_frame_instance.pack = MagicMock()
        mock_frame_instance.grid = MagicMock()

        # Mock ImageViewer's methods to do nothing but record calls
        ImageViewer._display_image = MagicMock()
        ImageViewer.title = MagicMock()
        ImageViewer.resizable = MagicMock()

        # Create an instance of ImageViewer
        viewer = ImageViewer(self.bird_db, start_idx=0)

        # Check that the attributes are set correctly
        self.assertIs(viewer.bird_db, self.bird_db)
        self.assertEqual(viewer.current_idx, 0)

        # Check that Tkinter mocks were used
        # Note: Since we replaced the tkinter module, we can't easily check the mock calls
        # but we can check that our mocks were used
        self.assertEqual(viewer.title.call_count, 1)
        self.assertEqual(viewer.resizable.call_count, 1)

        # Check that the labels and buttons were created
        self.assertGreaterEqual(mock_label_instance.pack.call_count, 1)  # image_label
        self.assertGreaterEqual(mock_button_instance.grid.call_count, 6)  # 6 navigation buttons
        self.assertGreaterEqual(mock_frame_instance.pack.call_count, 1) # btn_frame

    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_step_functions(self, MockFrame, MockButton, MockLabel):
        """Test the step functions by checking that they change current_idx correctly."""

        # Configure the ttk mocks
        mock_label_instance = MockLabel.return_value
        mock_label_instance.pack = MagicMock()
        mock_label_instance.configure = MagicMock()

        mock_button_instance = MockButton.return_value
        mock_button_instance.grid = MagicMock()

        mock_frame_instance = MockFrame.return_value
        mock_frame_instance.pack = MagicMock()
        mock_frame_instance.grid = MagicMock()

        # Mock ImageViewer's methods to do nothing but record calls
        ImageViewer._display_image = MagicMock()
        ImageViewer.title = MagicMock()
        ImageViewer.resizable = MagicMock()

        # Create an instance of ImageViewer
        viewer = ImageViewer(self.bird_db, start_idx=0)

        # Test initial state
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

        # Test _show_next_1 (should go to index 1)
        viewer._show_next_1()
        self.assertEqual(viewer.current_idx, 1)
        viewer._display_image.assert_called_once_with(1)
        viewer._display_image.reset_mock()

        # Test _show_previous_1 (should go back to index 0)
        viewer._show_previous_1()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

        # Test _show_next_10 (should go to index (0 + 10) % 3 = 1)
        viewer._show_next_10()
        self.assertEqual(viewer.current_idx, 1)
        viewer._display_image.assert_called_once_with(1)
        viewer._display_image.reset_mock()

        # Test _show_previous_10 (should go to index (1 - 10) % 3 = 0)
        viewer._show_previous_10()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

        # Test _show_next_100 (should go to index (0 + 100) % 3 = 1)
        viewer._show_next_100()
        self.assertEqual(viewer.current_idx, 1)
        viewer._display_image.assert_called_once_with(1)
        viewer._display_image.reset_mock()

        # Test _show_previous_100 (should go to index (1 - 100) % 3 = 0)
        viewer._show_previous_100()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

if __name__ == '__main__':
    unittest.main()
