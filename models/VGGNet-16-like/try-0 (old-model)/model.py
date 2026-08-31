import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from constants import NUM_OF_CLASSES

DTYPE = torch.bfloat16
IMAGE_SIZE = (224, 224)

class VGGNet16Like(nn.Module):
    def __init__(self, num_of_classes = NUM_OF_CLASSES):
        super(VGGNet16Like, self).__init__()

        self.input_dtype = DTYPE

        # Convolutional layers
        self.features = nn.Sequential(
            # shape = (B, 3, 224, 224)
            # ----------- block-1 -----------
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),

            nn.MaxPool2d(2),

            # shape = (B, 64, 112, 112)
            # ----------- block-2 -----------
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # shape = (B, 128, 56, 56)
            # ----------- block-3 -----------
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # shape = (B, 256, 28, 28)
            # ----------- block-4 -----------
            # Change from VGGNet-16 channels: 512 -> 256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # shape = (B, 256, 14, 14)
            # ----------- block-5 -----------
            # Change from VGGNet-16 channels: 512 -> 256
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # shape = (B, 256, 7, 7)
        ).to(DTYPE)

        self.classifier = nn.Sequential(
            # Change from VGGNet-16 linear layer size 4096 -> 1024
            nn.Linear(12544, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # Change from VGGNet-16 linear layer size 4096 -> 1024
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5), 

            # Change from VGGNet-16 linear layer size 1000 -> `num_of_classes`
            nn.Linear(1024, num_of_classes)
        ).to(DTYPE)


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
    model = VGGNet16Like()
    print(f"VGGNet16Like model created with {sum(p.numel() for p in model.parameters())} parameters")
