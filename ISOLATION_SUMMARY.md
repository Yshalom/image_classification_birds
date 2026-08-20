# Isolation of Database Handling

## Goal
Isolate the database loading and label extraction functionality into a separate package so it can be reused for other tasks (e.g., training a neural network).

## Changes Made

### Created New Package
- Created `src/bird_database/` package with:
  - `__init__.py` (empty)
  - `data_loader.py` - contains the `load_dataset` function (copied from `src/database_viewer/data_loader.py`)
  - `label_extractor.py` - contains the `extract_label_names_from_readme` function (copied from `src/database_viewer/label_extractor.py`)

### Updated Existing Modules
Modified the modules in `src/database_viewer/` to import from the new package for backward compatibility:

- `src/database_viewer/data_loader.py`:
  ```python
  from bird_database.data_loader import load_dataset
  __all__ = ["load_dataset"]
  ```

- `src/database_viewer/label_extractor.py`:
  ```python
  from bird_database.label_extractor import extract_label_names_from_readme
  __all__ = ["extract_label_names_from_readme"]
  ```

- `src/database_viewer/main.py`:
  Added code to include the `src` directory in the Python path so the `bird_database` package can be found:
  ```python
  import sys
  import os
  sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
  ```
  The imports from `data_loader` and `label_extractor` remain unchanged (they now import from the re-exported functions in the same directory, which in turn import from `bird_database`).

### Removed Duplicate File
- Removed `src/bird_database_viewer.py` as it was a duplicate of the original functionality and is no longer needed.

## Verification
- Created and ran a test script that successfully:
  - Loaded the dataset (shape: 2625 samples × 2 columns)
  - Extracted 526 label names from the README
- Ran the main application (with timeout) and confirmed that it successfully loads the dataset and extracts labels before attempting to initialize the GUI (which fails in a headless environment, but that is expected and unrelated to our changes).

## Result
The database handling is now isolated in the `bird_database` package and can be imported and used independently:
```python
from bird_database.data_loader import load_dataset
from bird_database.label_extractor import extract_label_names_from_readme

df = load_dataset("path/to/parquet")
label_names = extract_label_names_from_readme("path/to/README", label_name_path)
```

The existing `database_viewer` application continues to work unchanged from the user's perspective.