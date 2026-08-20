"""
Simple test to verify the refactored modules work correctly.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Try to import our modules
    from database_viewer.data_loader import load_dataset
    from database_viewer.label_extractor import extract_label_names_from_readme

    print("✓ Successfully imported modules")

    # Test data loading
    print("\nTesting data loading...")
    db_path = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
    df = load_dataset(db_path)
    print(f"✓ Loaded dataset with {len(df)} samples")
    print(f"✓ Columns: {list(df.columns)}")

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
    label_names = extract_label_names_from_readme(readme_path, label_name_path)
    print(f"✓ Extracted {len(label_names)} label names")
    print(f"✓ First 5 labels: {label_names[:5]}")

    print("\n✅ All tests passed! The refactored code is working correctly.")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()