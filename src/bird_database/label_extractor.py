"""
Label extractor module for the bird species image viewer.
Handles extracting label names from the README file.
"""


def extract_label_names_from_readme(readme_path: str, label_name_path: list[str]) -> list[str]:
    """
    Parse the README file to extract label names using a specific nested structure path.

    This function navigates through the README file by searching for each string in
    label_name_path in sequence, then extracts the indented list that follows,
    parsing it to get class ID to name mappings.

    Args:
        readme_path (str): Path to the README file
        label_name_path (list[str]): List of strings representing the nested path
                                   to the label names section (e.g., ["dataset_info:",
                                   "features:", "name: label", "dtype:", "class_label:", "names:"])

    Returns:
        list[str]: List of label names where index corresponds to class ID (0-based, sequential)

    Raises:
        Exception: If the README.md file structure is corrupted or labels cannot be extracted
    """
    # Read the entire README file
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        # Navigate through the nested structure by finding each section in order
        for section_name in label_name_path:
            # Find the line containing the current section name
            line_index = next(i for i, l in enumerate(lines) if section_name in l)
            # Calculate the indentation level of this section
            nested_level = lines[line_index].find(section_name)
            # Move to the lines after this section
            lines = lines[line_index + 1:]  # trim everything before and including this line

        # At this point, lines contains the content under the label_name_path section
        # Determine the indentation level of the first content line (should be the list)
        names_nested_level = next(i for i, c in enumerate(lines[0]) if c not in " \t")
        # Verify we actually found indented content (indicating a list)
        assert nested_level < names_nested_level, "The README.md doesn't contain labels names"

        # Find where the list ends by looking for a line with different indentation
        # This trims everything after the list of labels
        line_index = next(i for i, l in enumerate(lines) if
                          names_nested_level != next(j for j, c in enumerate(l) if c not in " \t"))
        lines = lines[:line_index]  # Keep only the list items

        # Extract class IDs and names from each line
        # Expected format: "  - 0: 'CLASS NAME'" or similar indented list format
        class_ids = (int(l[names_nested_level + 1:l.find(":") - 1]) for l in lines)
        class_names = (l[l.find(":") + 2:] for l in lines)

        # Create a dictionary mapping class ID to class name
        id_to_name_map = dict(zip(class_ids, class_names))

        # Build a list where index = class ID, assuming sequential IDs starting from 0
        # The length should match the number of lines we processed
        return [id_to_name_map[i] for i in range(len(lines))]

    except Exception as e:
        raise Exception("The README.md file structure is corrupted") from e