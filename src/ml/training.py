import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.EtherBits import EtherBits

from network import Network
from trainer import Trainer

path = "../../"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainSet = EtherBits(path, train=True, frame_size=1518, smallDataset=True)
testSet = EtherBits(path, train=False, frame_size=1518, smallDataset=True)

trainLoader = DataLoader(trainSet, batch_size=5, shuffle=True)
testLoader = DataLoader(testSet, batch_size=5, shuffle=True)

model = Network()

loss_fn = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainingManager = Trainer(model, loss_fn, 1E-3, trainLoader, testLoader, device)

trainingManager.train(1)

trainingManager.test()
