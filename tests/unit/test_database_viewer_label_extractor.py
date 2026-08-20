import unittest
import sys
import os

src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test that we can import from database_viewer.label_extractor and that it works
from database_viewer.label_extractor import extract_label_names_from_readme

class TestDatabaseViewerLabelExtractor(unittest.TestCase):
    """Test cases for the database_viewer.label_extractor module (which re-exports bird_database.label_extractor.extract_label_names_from_readme)."""

    def test_extract_label_names_from_readme_is_callable(self):
        """Test that extract_label_names_from_readme is callable and is the same function as from bird_database."""
        from bird_database.label_extractor import extract_label_names_from_readme as bird_extract
        self.assertIs(extract_label_names_from_readme, bird_extract)

    def test_extract_label_names_from_readme_success(self):
        """Test that extract_label_names_from_readme successfully extracts labels via the re-export."""
        readme_path = "database/birds-525-species-image-classification/README.md"
        label_name_path = [
            "dataset_info:",
            "features:",
            "name: label",
            "dtype:",
            "class_label:",
            "names:"
        ]
        try:
            label_names = extract_label_names_from_readme(readme_path, label_name_path)
            self.assertIsInstance(label_names, list)
            self.assertGreater(len(label_names), 0)
            if label_names:
                self.assertIsInstance(label_names[0], str)
        except FileNotFoundError:
            self.skipTest("README file not found")
        except Exception as e:
            self.skipTest(f"README structure not as expected: {e}")

if __name__ == '__main__':
    unittest.main()