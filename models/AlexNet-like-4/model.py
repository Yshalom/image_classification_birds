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
            # ----------- conv-1 -----------
            # shape = (B, 3, 224, 224)
            nn.Conv2d(3, 96, 11, 4),
            # shape = (B, 96, 54, 54)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # shape = (B, 96, 26, 26)
            
            # ----------- conv-2 -----------
            nn.Conv2d(96, 256, 5, padding=2),
            # shape = (B, 256, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # shape = (B, 256, 12, 12)
            
            # ----------- conv-3 -----------
            nn.Conv2d(256, 384, 3, padding=1),
            # shape = (B, 384, 12, 12)
            nn.ReLU(),

            # ----------- conv-4 -----------
            nn.Conv2d(384, 384, 3, padding=1),
            # shape = (B, 384, 12, 12)
            nn.ReLU(),

            # ----------- conv-5 -----------
            nn.Conv2d(384, 192, 3, padding=1),
            # shape = (B, 192, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # shape = (B, 192, 5, 5)
        ).to(DTYPE)

        self.classifier = nn.Sequential(
            # Change from AlexNet (6400, 4096) -> (4800, 512)
            nn.Linear(4800, 512, dtype=DTYPE),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # Change from AlexNet (4096, 4096) -> (512, 512)
            nn.Linear(512, 512, dtype=DTYPE),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # Change from AlexNet (512, 1000) -> (512, num_classes)
            nn.Linear(512, num_classes, dtype=DTYPE)
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
