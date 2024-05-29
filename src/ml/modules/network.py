import torch.nn.functional as F
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12144, 12144)
        self.fc2 = nn.Linear(12144, 12144)



    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
