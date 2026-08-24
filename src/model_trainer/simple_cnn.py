"""
Simple CNN model for bird species classification.
Designed to be very small for weak testing environments.
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from constants import NUM_OF_CLASSES

DTYPE = torch.bfloat16

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
            nn.Conv2d(3, 12, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 12, 46, 46)
            nn.Conv2d(12, 48, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 48, 22, 22)
            nn.Conv2d(48, 48, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # shape = (B, 48, 10, 10)
            nn.Conv2d(48, 48, kernel_size=3, dtype=DTYPE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )


        # shape = (B, 48, 4, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # shape = (B, 48*2*2)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(192, 96, dtype=DTYPE),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(96, 96, dtype=DTYPE),
            nn.ReLU(inplace=True),

            nn.Linear(96, num_classes, dtype=DTYPE)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = self.avgpool(x)
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
