"""Neural network models module."""

# Authors: Hamza Cherkaoui

import torch.nn as nn


class SimpleCIFARCNN(nn.Module):
    """Simple Convolutional Neural Network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleMLP(nn.Module):
    """Simple two-layer MLP."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
