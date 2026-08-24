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
IMAGE_SIZE = (224, 224)

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
            # shape = (B, 3, 224, 224)
            nn.Conv2d(3, 8, kernel_size=3, padding=1, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 8, 112, 112)
            nn.Conv2d(8, 8, kernel_size=3, padding=1, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 8, 56, 56)
            nn.Conv2d(8, 8, kernel_size=3, padding=1, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 8, 28, 28)
            nn.Conv2d(8, 8, kernel_size=3, padding=1, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 8, 14, 14)
            nn.Conv2d(8, 8, kernel_size=3, padding=1, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # shape = (B, 8*7*7)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(392, 128, dtype=DTYPE),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.3),
            nn.Linear(128, 128, dtype=DTYPE),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes, dtype=DTYPE)
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
    model = SimpleCNN(num_classes=10)
    print(f"SimpleCNN model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
