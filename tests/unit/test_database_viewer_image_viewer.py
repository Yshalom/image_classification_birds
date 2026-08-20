import unittest
import sys
import os
from unittest.mock import patch, MagicMock

src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# We'll try to import the ImageViewer class
try:
    from database_viewer.image_viewer import ImageViewer
    # We also need pandas for creating a dummy DataFrame
    import pandas as pd
    from PIL import Image
    import io
except ImportError as e:
    ImageViewer = None
    _import_error = e

@unittest.skipIf(ImageViewer is None, f"Unable to import ImageViewer: {_import_error}")
class TestImageViewer(unittest.TestCase):
    """Test cases for the database_viewer.image_viewer.ImageViewer class."""

    def setUp(self):
        """Set up a dummy DataFrame and other parameters for testing."""
        # Create a dummy DataFrame with the expected structure
        # The DataFrame should have a column "image" which is a dict with "bytes"
        # and a column "label" which is an integer.
        dummy_image_bytes = b"dummy image data"
        self.df = pd.DataFrame({
            "image": [{"bytes": dummy_image_bytes}, {"bytes": dummy_image_bytes}],
            "label": [0, 1]
        })
        self.indices = [0, 1]
        self.label_names = ["SPECIES_A", "SPECIES_B"]

        # We will patch the Tkinter dependencies to avoid needing a display
        # We patch tk.Tk and ttk.Label, etc., but note that the ImageViewer class
        # inherits from tk.Tk, so we need to mock the base class as well.
        # Instead, we can mock the entire tkinter module and ttk module.
        # However, a simpler approach is to patch the specific classes and methods
        # that are used in the __init__ and _display_image that require a display.

        # We'll start the patches in setUp and stop them in tearDown.

    def tearDown(self):
        pass

    @patch('database_viewer.image_viewer.tk.Tk')
    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_initialization(self, MockFrame, MockButton, MockLabel, MockTk):
        """Test that the ImageViewer can be initialized (with mocked Tkinter)."""
        # Configure the mocks to return mock objects
        mock_tk_instance = MockTk.return_value
        mock_tk_instance.title = MagicMock()
        mock_tk_instance.resizable = MagicMock()
        mock_tk_instance.bind = MagicMock()

        mock_label_instance = MockLabel.return_value
        mock_label_instance.pack = MagicMock()
        mock_label_instance.configure = MagicMock()

        mock_button_instance = MockButton.return_value
        mock_button_instance.grid = MagicMock()

        mock_frame_instance = MockFrame.return_value
        mock_frame_instance.pack = MagicMock()
        mock_frame_instance.grid = MagicMock()

        # Now try to create an instance of ImageViewer
        # We expect the __init__ to run without error (with our mocks)
        viewer = ImageViewer(self.df, self.indices, self.label_names, start_idx=0)

        # Check that the attributes are set correctly
        self.assertIs(viewer.df, self.df)
        self.assertEqual(viewer.indices, self.indices)
        self.assertEqual(viewer.label_names, self.label_names)
        self.assertEqual(viewer.current_idx, 0)

        # Check that the Tkinter mocks were used
        MockTk.assert_called_once()
        self.assertEqual(mock_tk_instance.title.call_count, 1)
        self.assertEqual(mock_tk_instance.resizable.call_count, 1)

        # Check that the labels and buttons were created
        # We expect at least the image_label and current_label to be created
        # and the buttons for navigation.
        # Note: The exact number of labels and buttons is fixed in the code.
        # We'll just check that the mocks were called a reasonable number of times.
        self.assertGreaterEqual(MockLabel.call_count, 2)  # image_label and current_label
        self.assertGreaterEqual(MockButton.call_count, 7)  # 7 navigation buttons
        self.assertGreaterEqual(MockFrame.call_count, 1)

    @patch('database_viewer.image_viewer.tk.Tk')
    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    @patch('database_viewer.image_viewer.Image.open')
    @patch('database_viewer.image_viewer.ImageTk.PhotoImage')
    def test_get_species_name(self, MockPhotoImage, MockImageOpen, MockFrame, MockButton, MockLabel, MockTk):
        """Test the _get_species_name method."""
        # We need to create an instance to test the method, but we can do so with mocks
        mock_tk_instance = MockTk.return_value
        mock_tk_instance.title = MagicMock()
        mock_tk_instance.resizable = MagicMock()
        mock_tk_instance.bind = MagicMock()

        mock_label_instance = MockLabel.return_value
        mock_label_instance.pack = MagicMock()
        mock_label_instance.configure = MagicMock()

        mock_button_instance = MockButton.return_value
        mock_button_instance.grid = MagicMock()

        mock_frame_instance = MockFrame.return_value
        mock_frame_instance.pack = MagicMock()
        mock_frame_instance.grid = MagicMock()

        viewer = ImageViewer(self.df, self.indices, self.label_names, start_idx=0)

        # Test known label IDs
        self.assertEqual(viewer._get_species_name(0), "SPECIES_A")
        self.assertEqual(viewer._get_species_name(1), "SPECIES_B")

        # Test out of range label ID
        self.assertEqual(viewer._get_species_name(2), "UNKNOWN CLASS 2")
        self.assertEqual(viewer._get_species_name(-1), "UNKNOWN CLASS -1")

    @patch('database_viewer.image_viewer.tk.Tk')
    @patch('database_viewer.image_viewer.ttk.Label')
    @patch('database_viewer.image_viewer.ttk.Button')
    @patch('database_viewer.image_viewer.ttk.Frame')
    def test_step_functions(self, MockFrame, MockButton, MockLabel, MockTk):
        """Test the step functions (_show_next_1, etc.) by checking that they change current_idx correctly."""
        # We need to mock the _display_image method to avoid trying to load images
        # and we also need to mock the Tkinter setup.

        mock_tk_instance = MockTk.return_value
        mock_tk_instance.title = MagicMock()
        mock_tk_instance.resizable = MagicMock()
        mock_tk_instance.bind = MagicMock()

        mock_label_instance = MockLabel.return_value
        mock_label_instance.pack = MagicMock()
        mock_label_instance.configure = MagicMock()

        mock_button_instance = MockButton.return_value
        mock_button_instance.grid = MagicMock()

        mock_frame_instance = MockFrame.return_value
        mock_frame_instance.pack = MagicMock()
        mock_frame_instance.grid = MagicMock()

        viewer = ImageViewer(self.df, self.indices, self.label_names, start_idx=0)

        # Mock the _display_image method to do nothing but record that it was called
        viewer._display_image = MagicMock()

        # Test initial state
        self.assertEqual(viewer.current_idx, 0)

        # Test _show_next_1 (should go to index 1)
        viewer._show_next_1()
        self.assertEqual(viewer.current_idx, 1)
        viewer._display_image.assert_called_once_with(1)

        # Reset the mock
        viewer._display_image.reset_mock()

        # Test _show_previous_1 (should go back to index 0)
        viewer._show_previous_1()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)

        # Test _show_next_10 (should go to index (0 + 10) % 2 = 0)
        viewer._show_next_10()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)

        # Test _show_previous_10 (should go to index (0 - 10) % 2 = 0 because -10 % 2 = 0)
        viewer._show_previous_10()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)

        # Test _show_next_100 (should go to index (0 + 100) % 2 = 0)
        viewer._show_next_100()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)

        # Test _show_previous_100 (should go to index (0 - 100) % 2 = 0)
        viewer._show_previous_100()
        self.assertEqual(viewer.current_idx, 0)
        viewer._display_image.assert_called_once_with(0)

if __name__ == '__main__':
    unittest.main()