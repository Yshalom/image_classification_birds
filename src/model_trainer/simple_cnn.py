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

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, dtype=DTYPE)
        self.conv2 = nn.Conv2d(12, 48, kernel_size=3, dtype=DTYPE)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, dtype=DTYPE)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # Assuming input image size will be pooled 3 times (divide by 8)
        # We'll use adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(256, 256, dtype=DTYPE) # 2*2*64 = 256
        self.fc2 = nn.Linear(256, num_classes, dtype=DTYPE)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Activation function
        self.inner_activation = nn.ELU()
        self.last_activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.
        """
        # Convert to DTYPE and normalize input
        x = x.type(DTYPE) / 255

        # Convolutional layers
        for conv_layer in (self.conv1, self.conv2, self.conv3):
            x = conv_layer(x)
            x = self.pool(x)
            x = self.inner_activation(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.inner_activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Convert to probabilities
        x = self.last_activation(x)
        
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
