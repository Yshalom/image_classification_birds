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

class AlexNetLike(nn.Module):
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
        super(AlexNetLike, self).__init__()

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
            # Change from AlexNet Conv2d(96, 256, 5, padding=2) -> Conv2d(64, 128, 5, padding=2)
            nn.Conv2d(96, 128, 5, padding=2),
            # shape = (B, 128, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # shape = (B, 128, 12, 12)
            
            # ----------- conv-3 -----------
            # Change from AlexNet Conv2d(192, 384, 3, padding=1) -> Conv2d(128, 192, 3, padding=1)
            nn.Conv2d(128, 192, 3, padding=1),
            # shape = (B, 192, 12, 12)
            nn.ReLU(),

            # ----------- conv-4 -----------
            # Change from AlexNet Conv2d(384, 384, 3, padding=1) ->Conv2d(192, 192, 3, padding=1)
            nn.Conv2d(192, 192, 3, padding=1),
            # shape = (B, 192, 12, 12)
            nn.ReLU(),

            # ----------- conv-5 -----------
            # Change from AlexNet Conv2d(384, 256, 3, padding=1) -> Conv2d(256, 128, 3, padding=1)
            nn.Conv2d(192, 128, 3, padding=1),
            # shape = (B, 128, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # shape = (B, 128, 5, 5)
        ).to(DTYPE)

        self.classifier = nn.Sequential(
            # Change from AlexNet (6400, 4096) -> (6400, 512)
            nn.Linear(3200, 512, dtype=DTYPE),
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
    model = AlexNetLike()
    print(f"SimpleCNN model created with {sum(p.numel() for p in model.parameters())} parameters")
