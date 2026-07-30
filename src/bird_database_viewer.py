# ==============================
# Constant declaration
# ==============================
# Path to the test portion of the bird species image database (parquet file)
DB_PATH = "database/birds-525-species-image-classification/data/test-00000-of-00001.parquet"
# Path to the README that contains the label-to-species mapping (used if present)
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
LABEL_NAME_PATH = ["dataset_info:", "features:", "name: label", "dtype:", "class_label:", "names:"]

# ==============================
# Imports
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
import tkinter as tk
from tkinter import ttk

# ==============================
# Load dataset and build label name mapping
# ==============================
# Read the parquet file into a DataFrame
df = pd.read_parquet(DB_PATH)

def extract_label_names_from_readme(readme_path: str) -> list[str]:
    """
    Parse the README file using the exact nested structure described by LABEL_NAME_PATH:
    """
    # Read the whole file – return an empty list if we cannot read it.
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        for section_name in LABEL_NAME_PATH:
            line_index = next(i for i, l in enumerate(lines) if section_name in l)
            nested_level = lines[line_index].find(section_name)
            lines = lines[line_index+1:] # trim boundary

        # Here the `lines` are the lines which under LABEL_NAME_PATH in the README.md file:
        
        names_nested_level = next(i for i, c in enumerate(lines[0]) if c not in " \t")
        assert nested_level < names_nested_level, "The README.md doesn't contain labels names"

        # trim boundary
        line_index = next(i for i, l in enumerate(lines) if 
                          names_nested_level != next(j for j, c in enumerate(l) if c not in " \t"))
        lines = lines[:line_index]

        # extract class's names and id
        class_ids = (int(l[names_nested_level+1:l.find(":")-1]) for l in lines)
        class_names = (l[l.find(":") + 2:] for l in lines)

        # place them into a list, where list[i] = class_name of class_id == i
        d = dict(zip(class_ids, class_names))
        return [d[i] for i in range(len(lines))]

    except Exception as e:
        raise "The README.md file structured is corrupted" from e

# Build the global mapping of label number to label name
# ------------------------------------------------------------------
# Extract the raw (index, name) tuples
LABEL_NAMES = extract_label_names_from_readme(README_PATH)

# --------------------------------
# GUI Application (Tkinter)
# ==============================
# A tiny GUI that shows images from the parquet file with navigation.
# --------------------------------
class ImageViewer(tk.Tk):
    """A tiny GUI that shows images from the parquet file with navigation."""

    def __init__(self, df, indices, start_idx=0):
        super().__init__()
        self.title("Bird Species Image Viewer")
        self.resizable(False, False)

        self.df = df
        self.indices = list(indices)          # list of valid row indices
        self.current_idx = start_idx

        # Label that will hold the image
        self.image_label = ttk.Label(self)
        self.image_label.pack(pady=10)

        # Frame for navigation buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)

        # ----- Backward navigation -------------------------------------------------
        # <<<  previous 100
        self.prev_100_button = ttk.Button(btn_frame, text="<<<", command=self._show_previous_100)
        self.prev_100_button.grid(row=0, column=0, padx=5)

        # <<  previous 10
        self.prev_10_button = ttk.Button(btn_frame, text="<<", command=self._show_previous_10)
        self.prev_10_button.grid(row=0, column=1, padx=5)

        # <  previous 1
        self.prev_1_button = ttk.Button(btn_frame, text="<", command=self._show_previous_1)
        self.prev_1_button.grid(row=0, column=2, padx=5)

        # current index display
        self.current_label = ttk.Label(btn_frame, text="", anchor="center")
        self.current_label.grid(row=0, column=4, padx=5)

        # ----- Forward navigation -------------------------------------------------
        # >  next 1
        self.next_button = ttk.Button(btn_frame, text=">", command=self._show_next_1)
        self.next_button.grid(row=0, column=5, padx=5)

        # >>  next 10
        self.next_10_button = ttk.Button(btn_frame, text=">>", command=self._show_next_10)
        self.next_10_button.grid(row=0, column=6, padx=5)

        # >>> next 100
        self.next_100_button = ttk.Button(btn_frame, text=">>>", command=self._show_next_100)
        self.next_100_button.grid(row=0, column=7, padx=5)

        # Bind keyboard shortcuts
        # Basic navigation
        self.bind("<Left>", lambda e: self._show_previous_1())
        self.bind("<Right>", lambda e: self._show_next_1())
        # Shift/Control modifiers
        self.bind("<Shift-Left>", lambda e: self._show_previous_10())
        self.bind("<Shift-Right>", lambda e: self._show_next_10())
        # Control modifies the step size to 100
        self.bind("<Control-Left>", lambda e: self._show_previous_100())
        self.bind("<Control-Right>", lambda e: self._show_next_100())
        # PageUp / PageDown for 100-step navigation (fallback if modifier keys not used)
        self.bind("<Next>", lambda e: self._show_previous_100())
        self.bind("<Prior>", lambda e: self._show_next_100())
        # Arrow keys for convenience
        self.bind("<Up>", lambda e: self._show_next_10())
        self.bind("<Down>", lambda e: self._show_previous_10())

        # Show the first image
        self._display_image(self.current_idx)

    def _display_image(self, idx: int):
        """Load the image at position idx, convert it for display, and update UI."""
        row = self.df.iloc[idx]
        pil_img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")

        # Resize for comfortable viewing while preserving aspect ratio
        max_w, max_h = 480, 480
        pil_img.thumbnail((max_w, max_h))

        self.photo = ImageTk.PhotoImage(pil_img)  # keep a reference!
        self.image_label.configure(image=self.photo)
        plt.close('all')  # close any stray matplotlib windows

        # Update window title with the bird name (using the mapped species name)
        label_id = row["label"]
        species_name = self._get_species_name(label_id)
        self.title(f"Bird Species - {species_name}")

        # Update the current-index label in the button bar
        self.current_label.configure(text=f"Index {self.current_idx + 1}/{len(self.indices)}")

    def _get_species_name(self, label_id: int) -> str:
        """
        Translate a numeric label to its species name using the tuple that is
        indexed by the class number. If the index is out of range we fall back
        to "UNKNOWN CLASS " + str(label_id)
        """
        try:
            return LABEL_NAMES[int(label_id)]
        except IndexError:
            return "UNKNOWN CLASS " + str(label_id)

    # ----- step functions -------------------------------------------------
    def _show_next_1(self):
        """Advance by a single step (default behavior)."""
        self._advance(1)

    def _show_previous_1(self):
        """Step backward by one."""
        self._advance(-1)

    def _show_next_10(self):
        """Advance by 10 steps (wrap-around)."""
        self._advance(10)

    def _show_previous_10(self):
        """Step backward by 10 steps (wrap-around)."""
        self._advance(-10)

    def _show_next_100(self):
        """Advance by 100 steps (wrap-around)."""
        self._advance(100)

    def _show_previous_100(self):
        """Step backward by 100 steps (wrap-around)."""
        self._advance(-100)

    def _advance(self, delta: int):
        """Internal helper that moves the current index by `delta` and refreshes."""
        self.current_idx = (self.current_idx + delta) % len(self.indices)
        self._display_image(self.current_idx)

# --------------------------------
# Entry point
# --------------------------------
if __name__ == "__main__":
    # All row indices in the parquet file
    all_indices = list(range(len(df)))
    viewer = ImageViewer(df, all_indices, start_idx=0)
    viewer.mainloop()
