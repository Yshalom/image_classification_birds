import unittest
import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from database_viewer.image_viewer import ImageViewer

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
        """Set up a dummy DataFrame and other parameters for testing."""
        # Create a dummy DataFrame with the expected structure
        # The DataFrame should have a column "image" which is a dict with "bytes"
        # and a column "label" which is an integer.
        self.df = pd.DataFrame({
            # The database has 2 rows
            "image": [{"bytes": DUMMY_IMAGE_BYTES}, {"bytes": DUMMY_IMAGE_BYTES}],
            "label": [0, 1]
        })
        self.label_names = ["SPECIES_A", "SPECIES_B"]

    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_initialization(self, MockFrame, MockButton, MockLabel):
        """Test that the ImageViewer can be initialized (with mocked objects)."""

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
        viewer = ImageViewer(self.df, self.label_names, start_idx=0)

        # Check that the attributes are set correctly
        self.assertIs(viewer.df, self.df)
        self.assertEqual(viewer.label_names, self.label_names)
        self.assertEqual(viewer.current_idx, 0)

        # Check that the Tkinter mocks were used
        self.assertEqual(viewer.title.call_count, 1) # _display_image is MagicMock therefore it doesn't call 'title'
        self.assertEqual(viewer.resizable.call_count, 1)

        # Check that the labels and buttons were created
        # We expect at least the image_label and current_label to be created
        # and the buttons for navigation.
        # Note: The exact number of labels and buttons is fixed in the code.
        # We'll just check that the mocks were called a reasonable number of times.
        self.assertGreaterEqual(MockLabel.call_count, 2)  # image_label and current_label
        self.assertGreaterEqual(MockButton.call_count, 6)  # 6 navigation buttons
        self.assertGreaterEqual(MockFrame.call_count, 1)

    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_get_species_name(self, MockFrame, MockButton, MockLabel):
        """Test the _get_species_name method."""

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
        viewer = ImageViewer(self.df, self.label_names, start_idx=0)

        # Test known label IDs
        self.assertEqual(viewer._get_species_name(0), "SPECIES_A")
        self.assertEqual(viewer._get_species_name(1), "SPECIES_B")

        # Test out of range label ID
        self.assertEqual(viewer._get_species_name(2), "UNKNOWN CLASS 2")
        self.assertEqual(viewer._get_species_name(-1), "UNKNOWN CLASS -1")

    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_step_functions(self, MockFrame, MockButton, MockLabel):
        """Test the step functions by checking that they change current_idx correctly."""

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
        viewer = ImageViewer(self.df, self.label_names, start_idx=0)

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

        # Test _show_next_10 (should go to index (0 + 10) % 2 = 0)
        viewer._show_next_10()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

        # Test _show_previous_10 (should go to index (0 - 10) % 2 = 0 because -10 % 2 = 0)
        viewer._show_previous_10()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

        # Test _show_next_100 (should go to index (0 + 100) % 2 = 0)
        viewer._show_next_100()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)
        viewer._display_image.reset_mock()

        # Test _show_previous_100 (should go to index (0 - 100) % 2 = 0)
        viewer._show_previous_100()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)

if __name__ == '__main__':
    unittest.main()
