import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
