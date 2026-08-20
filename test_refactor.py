"""
Test script to verify the refactored code works correctly.
This tests the data loading, label extraction, and GUI module imports without launching the GUI.
"""
import os
import sys

def main():
    """Run all tests."""
    print("=== Testing Refactored Bird Database Viewer ===\n")

    # Add src to Python path so we can import the package properly
    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, 'src')
    sys.path.insert(0, src_dir)

    # Import modules from the database_viewer package
    try:
        from database_viewer.data_loader import load_dataset
        print("✓ Successfully imported data_loader module")
    except Exception as e:
        print(f"✗ Failed to import data_loader module: {e}")
        return False

    try:
        from database_viewer.label_extractor import extract_label_names_from_readme
        print("✓ Successfully imported label_extractor module")
    except Exception as e:
        print(f"✗ Failed to import label_extractor module: {e}")
        return False

    try:
        from database_viewer.image_viewer import ImageViewer
        print("✓ Successfully imported image_viewer module")
    except Exception as e:
        print(f"✗ Failed to import image_viewer module: {e}")
        return False

    # Test data loading
    print("\nTesting data loading...")
    db_path = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
    try:
        df = load_dataset(db_path)
        print(f"✓ Successfully loaded dataset with {len(df)} samples")
        print(f"✓ Dataset columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Test label extraction
    print("\nTesting label extraction...")
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
        print(f"✓ Successfully extracted {len(label_names)} label names")
        print(f"✓ First few label names: {label_names[:5]}")
    except Exception as e:
        print(f"✗ Failed to extract label names: {e}")
        return False

    print("\n✅ All tests passed! The refactored code is working correctly.")
    print("\nNote: To run the full GUI application, you would need to:")
    print("1. Run in an environment with display capabilities")
    print("2. Execute: python3 -m src.database_viewer.main")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)