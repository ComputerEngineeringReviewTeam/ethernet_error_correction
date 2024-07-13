"""
Example of one the feed forward networks used in the project.
"""

import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)
