"""
Image viewer module for the bird species image viewer.
Contains the GUI application for viewing bird species images.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io


class ImageViewer(tk.Tk):
    """
    A GUI application that shows images from the parquet file with navigation.
    """

    def __init__(self, df, indices, label_names, start_idx=0):
        """
        Initialize the ImageViewer.

        Args:
            df (pd.DataFrame): DataFrame containing the bird species data
            indices (list): List of valid row indices to display
            label_names (list): List of label names where index corresponds to class ID
            start_idx (int): Starting index in the indices list
        """
        super().__init__()
        self.title("Bird Species Image Viewer")
        self.resizable(False, False)

        self.df = df
        self.indices = list(indices)          # list of valid row indices
        self.label_names = label_names        # list of label names
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
        """
        Load the image at position idx, convert it for display, and update UI.

        Args:
            idx (int): Index in the indices list to display
        """
        row = self.df.iloc[self.indices[idx]]
        pil_img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")

        # Resize for comfortable viewing while preserving aspect ratio
        max_w, max_h = 480, 480
        pil_img.thumbnail((max_w, max_h))

        self.photo = ImageTk.PhotoImage(pil_img)  # keep a reference!
        self.image_label.configure(image=self.photo)

        # Update window title with the bird name (using the mapped species name)
        label_id = row["label"]
        species_name = self._get_species_name(label_id)
        self.title(f"Bird Species - {species_name}")

        # Update the current-index label in the button bar
        self.current_label.configure(text=f"Index {self.current_idx + 1}/{len(self.indices)}")

    def _get_species_name(self, label_id: int) -> str:
        """
        Translate a numeric label to its species name using the label_names list.
        If the index is out of range we fall back to "UNKNOWN CLASS " + str(label_id)

        Args:
            label_id (int): Numeric label ID

        Returns:
            str: Species name corresponding to the label ID
        """
        try:
            return self.label_names[int(label_id)]
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
        """
        Internal helper that moves the current index by `delta` and refreshes.

        Args:
            delta (int): Number of steps to advance (can be negative)
        """
        self.current_idx = (self.current_idx + delta) % len(self.indices)
        self._display_image(self.current_idx)