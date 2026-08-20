import unittest
import sys
import os
from unittest.mock import patch, MagicMock

src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# We need to import the main function from database_viewer.main
try:
    from database_viewer.main import main
except ImportError as e:
    main = None
    _import_error = e

@unittest.skipIf(main is None, f"Unable to import main: {_import_error}")
class TestMain(unittest.TestCase):
    """Integration test for the main function in database_viewer.main."""

    @patch('database_viewer.main.ImageViewer')
    @patch('database_viewer.main.extract_label_names_from_readme')
    @patch('database_viewer.main.load_dataset')
    @patch('builtins.print')  # to capture print statements
    def test_main_success(self, mock_print, mock_load_dataset, mock_extract_label_names, mock_image_viewer):
        """Test that main runs successfully when all dependencies work."""
        # Set up the mocks
        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=100)  # so len(df) works
        mock_load_dataset.return_value = mock_df

        mock_label_names = ["label1", "label2"]
        mock_extract_label_names.return_value = mock_label_names

        mock_viewer_instance = MagicMock()
        mock_image_viewer.return_value = mock_viewer_instance

        # Call the main function
        main()

        # Check that load_dataset was called with the expected path
        mock_load_dataset.assert_called_once()
        args, kwargs = mock_load_dataset.call_args
        self.assertEqual(args[0], "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet")

        # Check that extract_label_names_from_readme was called with the expected arguments
        mock_extract_label_names.assert_called_once()
        args, kwargs = mock_extract_label_names.call_args
        self.assertEqual(args[0], "database/birds-525-species-image-classification/README.md")
        self.assertEqual(args[1], [
            "dataset_info:",
            "features:",
            "name: label",
            "dtype:",
            "class_label:",
            "names:"
        ])

        # Check that ImageViewer was instantiated with the correct arguments
        mock_image_viewer.assert_called_once()
        args, kwargs = mock_image_viewer.call_args
        self.assertIs(args[0], mock_df)  # the dataframe
        self.assertEqual(args[1], list(range(100)))  # all_indices (0 to 99)
        self.assertEqual(args[2], mock_label_names)  # label_names
        self.assertEqual(args[3], 0)  # start_idx
        # Check that no unexpected keyword arguments were passed
        self.assertEqual(kwargs, {})

        # Check that the viewer's mainloop method was called
        mock_viewer_instance.mainloop.assert_called_once()

        # Check that the expected print statements were made
        # We expect three prints: "Loading dataset...", "Dataset loaded with 100 samples",
        # "Extracting label names from README...", "Extracted 2 label names",
        # "Starting image viewer..."
        # Note: The exact strings are in the main function.
        expected_prints = [
            ("Loading dataset...", {}),
            (f"Dataset loaded with {len(mock_df)} samples", {}),
            ("Extracting label names from README...", {}),
            (f"Extracted {len(mock_label_names)} label names", {}),
            ("Starting image viewer...", {})
        ]
        # The mock_print call_args_list is a list of calls, each call is a tuple (args, kwargs)
        # We'll check that the expected prints are in the call_args_list (in order)
        actual_print_calls = [call.args for call in mock_print.call_args_list]
        self.assertEqual(len(actual_print_calls), len(expected_prints))
        for i, (expected_text, _) in enumerate(expected_prints):
            self.assertEqual(actual_print_calls[i][0], expected_text)

    @patch('database_viewer.main.ImageViewer')
    @patch('database_viewer.main.extract_label_names_from_readme')
    @patch('database_viewer.main.load_dataset')
    @patch('builtins.print')
    def test_main_load_dataset_failure(self, mock_print, mock_load_dataset, mock_extract_label_names, mock_image_viewer):
        """Test that main handles an exception from load_dataset."""
        # Set up load_dataset to raise an exception
        mock_load_dataset.side_effect = Exception("Test error")

        # Call main and expect it to propagate the exception
        with self.assertRaises(Exception) as context:
            main()

        self.assertEqual(str(context.exception), "Test error")

        # Check that the error was printed? Actually, the main function does not catch the exception,
        # so it will propagate. We don't expect any prints after the error, but we can check that
        # the initial print happened.
        mock_print.assert_any_call("Loading dataset...")
        # The other prints should not have been called
        mock_extract_label_names.assert_not_called()
        mock_image_viewer.assert_not_called()

    @patch('database_viewer.main.ImageViewer')
    @patch('database_viewer.main.extract_label_names_from_readme')
    @patch('database_viewer.main.load_dataset')
    @patch('builtins.print')
    def test_main_extract_label_names_failure(self, mock_print, mock_load_dataset, mock_extract_label_names, mock_image_viewer):
        """Test that main handles an exception from extract_label_names_from_readme."""
        # Set up load_dataset to return a dummy dataframe
        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_df

        # Set up extract_label_names_from_readme to raise an exception
        mock_extract_label_names.side_effect = Exception("Test error")

        # Call main and expect it to propagate the exception
        with self.assertRaises(Exception) as context:
            main()

        self.assertEqual(str(context.exception), "Test error")

        # Check that the expected prints up to the point of failure were made
        mock_print.assert_any_call("Loading dataset...")
        mock_print.assert_any_call(f"Dataset loaded with {len(mock_df)} samples")
        mock_print.assert_any_call("Extracting label names from README...")
        # The next print ("Extracted ...") should not have been called
        # Check that ImageViewer was not called
        mock_image_viewer.assert_not_called()

if __name__ == '__main__':
    unittest.main()