import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)
