"""
Image cache module for efficient training.
"""
import os
import sys
import torch

# Add src to path to import from database_reader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database_reader.bird_database import BirdDatabase
from constants import IMAGE_SIZE

class ImageCache:
    """
    A cache system for pre-loading and storing images as tensors to avoid
    repeated CPU-intensive image decoding during training/evaluation.
    """

    def __init__(self,
                 database: BirdDatabase,
                 image_size: tuple[int, int] = IMAGE_SIZE,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the ImageCache by pre-loading all images from the database.

        Args:
            database: BirdDatabase instance to load images from
            image_size: Target size for images (width, height)
            device: Target device for tensors (defaults to CPU for caching)
        """
        self.database = database
        self.image_size = image_size
        self.device = device

        # Pre-load all images and convert to tensors
        print(f"Pre-loading {len(database)} images into cache...")
        self._image_tensors = []
        self._labels = []

        for idx in range(len(database)):
            # Get image as numpy array and convert to tensor
            img_array = database.get_img_np(idx, image_size)  # shape=(height, width, 3)
            img_array = img_array.transpose(2, 0, 1)          # shape=(3, height, width)
            img_tensor = torch.from_numpy(img_array)
            self._image_tensors.append(img_tensor)

            # Get label
            label = database.get_id(idx)
            self._labels.append(label)

            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"  Loaded {idx + 1}/{len(database)} images")

        # Stack all tensors for efficient access
        self._image_tensors = torch.stack(self._image_tensors).to(torch.uint8).to(self.device)
        self._labels = torch.tensor(self._labels, dtype=torch.long, device=self.device)

        print(f"Image cache created: {self._image_tensors.shape} images, {len(self._labels)} labels")

    def __getitem__(self, key: int | slice | tuple[int]) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, int]:
        """
        Get cached image tensor and label(s) by index, slice, or iterable of indices.
        Supports: image_cache[idx], image_cache[start:stop:step], image_cache[list_of_indices]

        Args:
            key: Integer index, slice object, or iterable of indices

        Returns:
            If key is int: Tuple of (image_tensor, label)
            If key is slice or iterable: Tuple of (image_batch_tensor, label_batch_tensor)
        """
        if isinstance(key, int):
            return self._image_tensors[key], self._labels[key].item()
        elif isinstance(key, slice) \
            or (isinstance(key, torch.Tensor) and len(key.shape) == 1 and key.shape[0] < len(self) and key.dtype == torch.long):
            # Assume iterable of indices
            return self._image_tensors[key], self._labels[key]
        else:
            raise IndexError(f"The `key` must be either `int` or `slice` or `indices-tensor` - Got type `{type(key)}`")

    def __len__(self) -> int:
        return len(self._image_tensors)
