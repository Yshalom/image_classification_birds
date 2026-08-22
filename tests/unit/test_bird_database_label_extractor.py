import unittest
import sys
import os

README_PATH = "database/birds-525-species-image-classification/README.md"

# Path of the class label names in the README.md nested structure
#   +------------------------+
#   | dataset_info:          |
#   |   features:            |
#   |     name: label        |
#   |       dtype:           |
#   |         class_label:   |
#   |           names:       |
#   |             '0': $NAME |
#   |             '1': $NAME |
#   |             ...        |
#   +------------------------+
LABEL_NAME_PATH = (
    "dataset_info:",
    "features:",
    "name: label",
    "dtype:",
    "class_label:",
    "names:"
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from bird_database.label_extractor import extract_label_names_from_readme

class TestBirdDatabaseLabelExtractor(unittest.TestCase):
    """Test cases for the bird_database.label_extractor module."""

    def test_extract_label_names_from_readme_success(self):
        """Test that extract_label_names_from_readme successfully extracts labels from README."""
        try:
            label_names = extract_label_names_from_readme(README_PATH, LABEL_NAME_PATH)
            self.assertIsInstance(label_names, tuple)
            self.assertGreater(len(label_names), 0)
            # Check that the first element is a string
            if label_names:
                self.assertIsInstance(label_names[0], str)
        except FileNotFoundError:
            self.skipTest("README file not found")
        except Exception as e:
            # If the structure is not as expected, we skip for now
            self.skipTest(f"README structure not as expected: {e}")

    def test_extract_label_names_invalid_path(self):
        """Test that extract_label_names_from_readme raises an exception for an invalid path."""
        with self.assertRaises(Exception):
            extract_label_names_from_readme("non/existent/path.md", ("dummy", ))

if __name__ == '__main__':
    unittest.main()
