"""
Simple CNN model for bird species classification.
Designed to be very small for weak testing environments.
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from constants import NUM_OF_CLASSES

DTYPE = torch.bfloat16
IMAGE_SIZE = (94, 94)

class SimpleCNN(nn.Module):
    """
    A simple CNN for image classification with minimal layers.
    Suitable for weak testing environments as requested.
    """

    def __init__(self, num_classes = NUM_OF_CLASSES):
        """
        Initialize the SimpleCNN.

        Args:
            num_classes (int): Number of output classes (default: 525 for bird species)
        """
        super(SimpleCNN, self).__init__()

        self.input_dtype = DTYPE

        # Convolutional layers
        self.features = nn.Sequential(
            # shape = (B, 3, 94, 94)
            nn.Conv2d(3, 24, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 24, 46, 46)
            nn.Conv2d(24, 24, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 24, 22, 22)
            nn.Conv2d(24, 24, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 24, 10, 10)
            nn.Conv2d(24, 24, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # shape = (B, 24*4*4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(384, 96, dtype=DTYPE),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3),
            nn.Linear(96, 96, dtype=DTYPE),
            nn.ReLU(inplace=True),

            nn.Linear(96, num_classes, dtype=DTYPE)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Simple test
    model = SimpleCNN()
    print(f"SimpleCNN model created with {sum(p.numel() for p in model.parameters())} parameters")
