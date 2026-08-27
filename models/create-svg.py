"""
CSV -> SVG plotter with averaging

Expected CSV format (first line = header):
    training loops, loss(DB-train), loss(DB-test), loss(DB-validation)
where the first column is the X-axis,
and subsequent columns are Y-axis data to be averaged across files.

For each Y-field, produces an SVG file showing:
- Individual series (one per CSV file) in different colors, points + lines (no labels)
- Average series in bold gray, points + lines + (x,y) labels near points

Output files:
    <basename>-<graph_name>.svg
where ``graph_name`` comes from ``GRAPH_NAME_MAP``.
"""

import csv
import os
import sys
import glob
import argparse
from numbers import Number
from typing import Iterable

# Color constants for terminal output
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
GRAPH_COLOR_MAP = {
    "loss(DB-train)": ("#ff0000", "#ff000077"),
    "loss(DB-test)": ("#00ff00", "#00ff0077"),
    "loss(DB-validation)": ("#0000ff", "#0000ff77")
}
# Mapping from canonical y-field name to the name used for the SVG file
GRAPH_NAME_MAP = {
    "loss(DB-train)": "train",
    "loss(DB-test)": "test",
    "loss(DB-validation)": "validation",
    # extend here for additional y-fields
}
VALID_Y_AXIS = list(GRAPH_NAME_MAP.keys())

# SVG canvas size
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800

# Padding around the plot area (pixels)
PADDING_LEFT = 80
PADDING_RIGHT = 40
PADDING_TOP = 60
PADDING_BOTTOM = 60

# Axis & grid appearance
NUM_TICKS = 15
TICK_SIZE = 4
GRID_STROKE_WIDTH = 1
GRID_COLOR = "#ddd"
AXIS_STROKE_WIDTH = 2
AXIS_COLOR = "#777"
AXIS_MARGIN_PERCENTAGE = .03

# Point appearance
POINT_RADIUS = 2
AVERAGE_POINT_RADIUS = 3

# Line appearance
LINE_WIDTH = 1
AVERAGE_LINE_WIDTH = 2

# Text appearance
IMAGE_TITLE = "Training Graph"
LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 14
LABEL_SPACING_VERT = 20
LABEL_OFFSET_FROM_TOP = PADDING_TOP // 2
LABEL_OFFSET_FROM_RIGHT = 10
TITLE_FONT_SIZE = 24
TITLE_OFFSET_FROM_TOP = 30
TITLE_COLOR = "blue"
POINT_COORDINATES_LABEL_FONT_SIZE = 12
POINT_COORDINATES_LABEL_X_OFFSET = -30
POINT_COORDINATES_LABEL_Y_OFFSET = 4
POINT_COORDINATES_LABEL_Y_INDEX_OFFSET = 20

# SVG overall layout
SVG_BACKGROUND = "black"
SVG_HEADER_STR = f'<svg width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" xmlns="http://www.w3.org/2000/svg" version="1.1">'
SVG_TAIL_STR = "</svg>"

# ----------------------------------------------------------------------
# Helper: format numbers with at least 4 digit precision
# ----------------------------------------------------------------------
def format_val(v: float) -> str:
    """Format a number with at least 4 digit precision.
    - If the absolute value has 4+ digits (>=1000), render as integer.
    - Otherwise, format with the appropriate number of decimal places to keep at least 4 significant digits, removing trailing zeros.
    """
    # Handle zero explicitly to avoid issues with digit counting
    if v == 0:
        return "0"
    # Count digits before the decimal point in the absolute value
    digits_before = len(str(int(abs(v))))
    # Determine how many decimal places we need to reach at least 4 significant digits
    decimals_needed = max(0, 4 - digits_before)
    # Round to that many decimal places
    rounded = round(v, decimals_needed)
    # Format with fixed-point
    s = f"{rounded:.{decimals_needed}f}"
    # Remove trailing zeros and a trailing decimal point if any
    return s.rstrip('0').rstrip('.')

# ----------------------------------------------------------------------
# CSV reading & preprocessing for single file
# ----------------------------------------------------------------------
def read_single_csv(filepath: str) -> \
    tuple[
        tuple[int],                     # x_values
        dict[str, dict[int, float]],  # data: { y_field: { x -> y } }
    ]:
    """
    Read a single CSV file.
    Returns:
        - x_values: tuple of x values (int) in order of appearance
        - data: dict mapping each y-field name to a dict {x -> y}
    """
    # open and read the CSV file
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
            rows = list(reader)
        except StopIteration:
            raise ValueError("CSV file is empty.")

    # Basic header validation
    if not header:
        raise ValueError("CSV header is empty.")
    # ---- discover y-field columns -----------------------------------------
    # For our format: header[0] is x-axis, rest are y-field names like "loss(DB-train)"
    y_values_in_header = header[1:]  # All columns except the first (x-axis)

    # Map y-field names to their column indices
    yfield_to_col_indices: dict[str, int] = {}
    for idx, col_name in enumerate(y_values_in_header, start=1):  # start=1 because we skipped x-axis
        if col_name in VALID_Y_AXIS:
            yfield_to_col_indices[col_name] = idx
        else:
            # ignore columns we don't have a mapping for
            continue

    if not yfield_to_col_indices:
        raise ValueError("No columns matched known y-field names.")

    # ---- parse rows ---------------------------------------------------------
    # Prepare storage: for each y-field, a dict {x -> y}
    # For single file processing, we use file_index = 0
    data: dict[str, dict[int, float]] = {
        y_field: {} for y_field in yfield_to_col_indices.keys()
    }

    x_values: list[int] = []

    for row_num, row in enumerate(rows, start=1):
        # Skip empty/malformed lines
        if (not row or all(cell.strip() == '' for cell in row)) or len(row) < len(header):
            print(f"{YELLOW}Warning: line {row_num} has fewer columns than header - skipping.{RESET}")
            continue

        # Parse x value (first column)
        try:
            x = int(row[0])
        except ValueError:
            print(f"{YELLOW}Warning: cannot parse X value on line {row_num} - skipping.{RESET}")
            continue
        x_values.append(x)

        # Parse each y-field's columns
        for y_field, col_idx in yfield_to_col_indices.items():
            try:
                val = float(row[col_idx])
            except (ValueError, IndexError):
                print(f"{YELLOW}Warning: bad numeric value for {y_field} on line {row_num} - skipping this row.{RESET}")
                # Abort this row entirely – we don't have complete data for it
                break
            # For single file, we use file_index = 0
            data[y_field][x] = val

    # Convert to immutable tuples for the caller
    return tuple(x_values), data


# ----------------------------------------------------------------------
# Multiple CSV reading & averaging
# ----------------------------------------------------------------------
def read_and_average_csvs(directory: str) -> \
    tuple[
        tuple[str],                         # filenames
        tuple[int],                         # x_values
        dict[str, tuple[dict[int, float]]], # series data: {y_field: tuple[{ x -> y}] }
                # for each y_field, multiple functions `x -> y` each one for a file.
        dict[str, dict[int, float]],        # averaged data: {y_field: { x -> y } }
                # for each y_field, a function `x -> y`
    ]:
    """
    Read all CSV files in directory and compute averaged y-values across files at same x-position.

    Returns:
        - filenames: tuple of filenames (str) in the same order as the `series_data`'s tuples.
        - x_values: tuple of x values (int) in order of appearance (assumed consistent across files)
        - series_data: dict mapping each y-field to tuple of dicts { x -> y }
                       each tuple is the values of a file.
        - averaged_data: dict mapping each y-field to dict { x -> y }
                         containing the averaged y values for each x position across all files.
    """

    # Find all CSV files in directory
    csv_pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")

    csv_files.sort()  # For deterministic ordering

    # Store data from each file:
    # filename -> { y_field -> { x -> y } }
    file_data: dict[str, dict[str, dict[int, float]]] = {}

    # Read each CSV file
    x_values = None
    y_field_names = None
    filenames = []
    for csv_file in csv_files:
        # Read single file
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        x_values_single, file_y_data = read_single_csv(csv_file)
        file_data[filename] = file_y_data
        filenames.append(filename)

        # Verify all files have same x_values
        if x_values is None:
            x_values = x_values_single
        elif x_values != x_values_single:
            raise ValueError(f"Inconsistent x-values between files: {csv_files[0]} and {csv_file}")

        # Verify all files have same y_fields
        if y_field_names is None:
            y_field_names = set(file_y_data.keys())
        elif y_field_names != set(file_y_data.keys()):
            raise ValueError(f"Inconsistent y-fields between files: {csv_files[0]} and {csv_file}")

    x_values = tuple(x_values)
    y_field_names = tuple(y_field_names)
    filenames = tuple(filenames)

    # Prepare return data structures
    series_data: dict[str, dict[(str, int), float]] = {}  # { y_field -> tuple[{ x -> y }] }
    averaged_data: dict[str, dict[int, float]] = {}       # { y_field -> { x -> y } }

    for y_field in y_field_names:
        series_data[y_field] = tuple({} for _ in range(len(filenames)))
        averaged_data[y_field] = {}
        for x in x_values:
            avg = 0
            for idx, filename in enumerate(filenames):
                val = file_data[filename][y_field][x]
                series_data[y_field][idx][x] = val
                avg += val
            averaged_data[y_field][x] = avg / len(filenames)

    return filenames, x_values, series_data, averaged_data


# ----------------------------------------------------------------------
# Scaling helpers
# ----------------------------------------------------------------------

def make_scaling(x_min: Number, x_max: Number, y_min: Number, y_max: Number):
    """Return scaling-x, scaling-y functions for the given limits.
    The scaling functions map a value in [min, max] to the SVG canvas's axis.
    """

    def map_value(v, v_min, v_max, out_min, out_max):
        """Map v from [v_min, v_max] -> [out_min, out_max] (linear)."""
        if v_max == v_min:          # avoid division by zero
            return (out_min + out_max) / 2.0
        return out_min + (v - v_min) * (out_max - out_min) / (v_max - v_min)
    
    def sx(x: Number) -> Number:
        return map_value(x, x_min, x_max, PADDING_LEFT, CANVAS_WIDTH - PADDING_RIGHT)

    def sy(y: Number) -> Number:
        return map_value(y, y_min, y_max, CANVAS_HEIGHT - PADDING_BOTTOM, PADDING_LEFT)
    
    return sx, sy

def nice_ticks(v_min: Number, v_max: Number, steps: int = NUM_TICKS) -> list[Number]:
    """Return `steps` evenly spaced values from v_min to v_max (inclusive)."""
    if steps < 2:
        steps = 2
    step = (v_max - v_min) / (steps - 1)
    return [v_min + i * step for i in range(steps)]

def compute_limits(axis_vals: Iterable[Number]) -> tuple[Number, Number]:
    """
    Return (min, max) with a margin.
    `axis_vals` are iterables of numbers.
    """
    if not axis_vals:
        raise ValueError("SVG limit computation: No data to compute limits.")

    min_val = min(axis_vals)
    max_val = max(axis_vals)

    # margin
    axis_range = max_val - min_val
    margin = AXIS_MARGIN_PERCENTAGE * axis_range

    return min_val - margin, max_val + margin

# ----------------------------------------------------------------------
# SVG drawing helpers
# ----------------------------------------------------------------------

def draw_grid_and_ticks(svg_parts: list[str], x_min: Number, x_max: Number, y_min: Number, y_max: Number, sx, sy) -> None:
    """Add grid lines, tick marks and numeric axis labels."""

    xticks = nice_ticks(x_min, x_max, NUM_TICKS)
    yticks = nice_ticks(y_min, y_max, NUM_TICKS)

    # Vertical grid lines as a single path
    v_grid_paths = []
    v_tick_paths = []
    for x_val in xticks:
        x_svg = sx(x_val)
        # grid line
        v_grid_paths.append(f"{x_svg:.2f},{sy(y_min):.2f} {x_svg:.2f},{sy(y_max):.2f}")
        # tick mark (below axis)
        v_tick_paths.append(f"{x_svg:.2f},{sy(y_min):.2f} {x_svg:.2f},{sy(y_min) + TICK_SIZE:.2f}")

    if v_grid_paths:
        svg_parts.append(
            f'  <path d="M {" M ".join(v_grid_paths)}" '
            f'stroke="{GRID_COLOR}" stroke-width="{GRID_STROKE_WIDTH}" fill="none"/>'
        )
    if v_tick_paths:
        svg_parts.append(
            f'  <path d="M {" M ".join(v_tick_paths)}" '
            f'stroke="{AXIS_COLOR}" stroke-width="1" fill="none"/>'
        )

    # x-axis labels
    for x_val in xticks:
        x_svg = sx(x_val)
        svg_parts.append(
            f'  <text x="{x_svg:.2f}" y="{sy(y_min) + TICK_SIZE + 12:.2f}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="{TICK_LABEL_FONT_SIZE}" fill="{AXIS_COLOR}">{format_val(x_val)}</text>'
        )

    # Horizontal grid lines as a single path
    h_grid_paths = []
    h_tick_paths = []
    for y_val in yticks:
        y_svg = sy(y_val)
        # grid line
        h_grid_paths.append(f"{sx(x_min):.2f},{y_svg:.2f} {sx(x_max):.2f},{y_svg:.2f}")
        # tick mark (left of axis)
        h_tick_paths.append(f"{sx(x_min) - TICK_SIZE:.2f},{y_svg:.2f} {sx(x_min):.2f},{y_svg:.2f}")

    if h_grid_paths:
        svg_parts.append(
            f'  <path d="M {" M ".join(h_grid_paths)}" '
            f'stroke="{GRID_COLOR}" stroke-width="{GRID_STROKE_WIDTH}" fill="none"/>'
        )
    if h_tick_paths:
        svg_parts.append(
            f'  <path d="M {" M ".join(h_tick_paths)}" '
            f'stroke="{AXIS_COLOR}" stroke-width="1" fill="none"/>'
        )

    # y-axis labels
    for y_val in yticks:
        y_svg = sy(y_val)
        svg_parts.append(
            f'  <text x="{sx(x_min) - TICK_SIZE - 4:.2f}" y="{y_svg + 4:.2f}" '
            f'text-anchor="end" font-family="sans-serif" '
            f'font-size="{TICK_LABEL_FONT_SIZE}" fill="{AXIS_COLOR}">{format_val(y_val)}</text>'
        )

def draw_axes(svg_parts: list[str], x_min: Number, x_max: Number, y_min: Number, y_max: Number, sx, sy) -> None:
    """Draw the main X and Y axes (over the grid/ticks)."""
    svg_parts.append(
        f'  <path d="M {sx(x_min):.2f},{sy(y_min):.2f} L {sx(x_max):.2f},{sy(y_min):.2f} M {sx(x_min):.2f},{sy(y_min):.2f} L {sx(x_min):.2f},{sy(y_max):.2f}" '
        f'stroke="{AXIS_COLOR}" stroke-width="{AXIS_STROKE_WIDTH}" fill="none"/>'
    )

def draw_series(svg_parts: list[str],
                x_values: tuple[int],
                series: dict[int, float],
                color,
                sx, sy) -> None:
    """
    Plot individual series: points + connecting lines (no labels).
    Each CSV file gets a different color from COLOR_LIST.
    """

    points = []  # Store (x, y) coordinates for this series

    # Collect points for this series in order of x values
    for x in x_values:
        # Get the y-value for this file at this x position
        y = series[x]
        cx, cy = sx(x), sy(y)
        points.append((cx, cy))

        # Draw the point
        svg_parts.append(
            f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{POINT_RADIUS}" '
            f'fill="{color}"/>'
        )

    # Draw lines connecting consecutive points as a single polyline
    if len(points) > 1:
        points_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
        svg_parts.append(
            f'  <polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="{LINE_WIDTH}"/>'
        )

def draw_average_series(svg_parts: list[str],
                        x_values: tuple[int],
                        averaged_series_data: dict[int, float],
                        color,
                        sx, sy) -> None:
    """
    Plot average series: points + connecting lines + (x,y) labels near points.
    Uses bold gray line.
    """
    # Average series gets special styling

    points = []  # Store (x, y) coordinates for this series

    # Collect points for average series in order of x values
    for i, x in enumerate(x_values):
        y = averaged_series_data.get(x)
        if y is not None:
            cx, cy = sx(x), sy(y)
            points.append((cx, cy))

            # Draw the point
            svg_parts.append(
                f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{AVERAGE_POINT_RADIUS}" '
                f'fill="{color}"/>'
            )

            # Add coordinate label above the point
            label_cx = cx + POINT_COORDINATES_LABEL_X_OFFSET
            label_cy = cy + POINT_COORDINATES_LABEL_Y_OFFSET + POINT_COORDINATES_LABEL_Y_INDEX_OFFSET * (i % 2 * 2 - 1)
            svg_parts.append(
                f'  <text x="{label_cx}" y="{label_cy}" '
                f'font-family="sans-serif" font-size="{POINT_COORDINATES_LABEL_FONT_SIZE}" fill="{color}">({x}, {format_val(y)})</text>'
            )

    # Draw lines connecting consecutive points as a single polyline
    if len(points) > 1:
        points_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in points])
        svg_parts.append(
            f'  <polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="{AVERAGE_LINE_WIDTH}"/>'
        )

def draw_label(svg_parts: list[str], label: str, label_x, label_y, color):
    """Add series labels in the top-right corner."""
    svg_parts.append(
        f'  <text x="{label_x}" y="{label_y}" '
        f'font-family="sans-serif" font-size="{LABEL_FONT_SIZE}" '
        f'fill="{color}">{label}</text>')

def draw_rectangle(svg_parts: list[str], width, height, color) -> None:
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="{color}"/>')

def draw_title(svg_parts: list[str], title: str, title_x, title_y, color) -> None:
    svg_parts.append(
        f'  <text x="{title_x}" y="{title_y}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="{TITLE_FONT_SIZE}" fill="{color}">{title}</text>')


# ----------------------------------------------------------------------
# SVG creation functions
# ----------------------------------------------------------------------

def write_svg(filepath: str, svg_parts: list[str]) -> None:
    """Write the accumulated SVG lines to disk."""
    try:
        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(svg_parts))
        print(f"SVG written to: {filepath}")
    except OSError as e:
        print(f"Failed to write SVG file: {e}")
        sys.exit(1)

def create_svg_with_average(filenames: tuple[str],
                            x_values: tuple[int],
                            series_data: dict[str, tuple[dict[int, float]]],
                            averaged_data: dict[str, dict[int, float]],
                            output_filename: str) -> None:
    """
    Create SVG file showing both individual series and averaged data.
    For each y-field, draw the graph:
    - Individual series: points + lines (no labels)
    - Average series: in bold points + lines + (x,y) labels
    """

    # Compute limits based on both individual and averaged data
    all_y_values = []

    # Add individual series values
    for series in series_data.values():
        for vals in series:
            all_y_values.extend(vals.values())

    # Add averaged series values
    for vals in averaged_data.values():
        all_y_values.extend(vals.values())

    if not all_y_values:
        raise ValueError("No data to plot")

    # Define the scaling functions
    y_min, y_max = compute_limits(all_y_values)
    x_min, x_max = compute_limits(x_values)
    scaling_x_func, scaling_y_func = make_scaling(x_min, x_max, y_min, y_max)

    svg_parts: list[str] = []
    svg_parts.append(SVG_HEADER_STR)

    # ----------------------------------------------------------------
    #                    Start drawing the image
    # ----------------------------------------------------------------
    if SVG_BACKGROUND:
        draw_rectangle(svg_parts, CANVAS_WIDTH, CANVAS_HEIGHT, SVG_BACKGROUND) # Draw background

    # Draw title, grid ticks and axes
    draw_title(svg_parts, IMAGE_TITLE, CANVAS_WIDTH / 2, TITLE_OFFSET_FROM_TOP, TITLE_COLOR)
    draw_grid_and_ticks(svg_parts, x_min, x_max, y_min, y_max, scaling_x_func, scaling_y_func)
    draw_axes(svg_parts, x_min, x_max, y_min, y_max, scaling_x_func, scaling_y_func)

    # Coordinates of the labels of subgraphs series (top-left), spaced from each other on the y-axis
    label_y = LABEL_OFFSET_FROM_TOP
    label_x = PADDING_LEFT + LABEL_OFFSET_FROM_RIGHT

    # Process each y-field
    for y_field, y_series_data in series_data.items():
        if y_field not in GRAPH_NAME_MAP:
            continue

        # Get data for this y-field
        y_average_data = averaged_data[y_field]   # Averaged series data: {x: y_avg}
        average_color, series_color = GRAPH_COLOR_MAP[y_field]

        # Draw the label
        draw_label(svg_parts, GRAPH_NAME_MAP[y_field], label_x, label_y, average_color)
        label_y += LABEL_SPACING_VERT

        # For all the sub-graphs
        for subseries_data in y_series_data:        
            # Plot individual series (points + lines, no labels)
            draw_series(svg_parts, x_values, subseries_data, series_color, scaling_x_func, scaling_y_func)

        # Plot average series (points + lines + labels)
        draw_average_series(svg_parts, x_values, y_average_data, average_color, scaling_x_func, scaling_y_func)

    svg_parts.append(SVG_TAIL_STR)
    write_svg(output_filename, svg_parts)

def plot_to_svg(csv_dir_filepath: str) -> None:
    """
    Create SVG files showing both individual series and averaged data from all CSV files in the same directory.
    """

    # Read and average all CSV files in the directory
    try:
        filenames, x_values, series_data, averaged_data = read_and_average_csvs(csv_dir_filepath)
    except Exception as e:
        # Fall back to single file processing if directory reading fails
        raise ValueError(f"Could not read directory for averaging: ") from e

    # Create SVGs with both individual and average series
    csv_filename = os.path.join(csv_dir_filepath, 'graph.svg')
    create_svg_with_average(filenames, x_values, series_data, averaged_data, csv_filename)

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to SVG plots with averaging.')
    parser.add_argument('root_dir', nargs='?', default='.', help='Root directory to search for CSV files (default: current directory)')
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)

    # Check if root_dir contains CSV files directly
    csv_files_in_root = [f for f in os.listdir(root_dir) if f.lower().endswith('.csv')]

    if csv_files_in_root:
        # Process the root directory directly
        print(f"Processing directory: {root_dir}")
        # Use any CSV file from the directory to get the directory path
        plot_to_svg(os.path.join(root_dir, csv_files_in_root[0]))
        print("--- SVG images were created")
        return

    # Walk the directory tree to find directories containing CSV files
    dirs_to_process = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.lower().endswith('.csv') for f in filenames):
            dirs_to_process.append(dirpath)

    # Process each directory containing CSV files
    for dir_path in sorted(dirs_to_process):
        print(f"Processing directory: {dir_path}")
        plot_to_svg(dir_path)
        print("--- SVG images were created")

if __name__ == "__main__":
    main()
