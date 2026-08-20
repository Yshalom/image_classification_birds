"""
Label extractor module for the bird species image viewer.
Handles extracting label names from the README file.
"""

from bird_database.label_extractor import extract_label_names_from_readme

# Re-export for backward compatibility
__all__ = ["extract_label_names_from_readme"]