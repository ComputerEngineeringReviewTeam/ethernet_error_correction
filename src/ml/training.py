import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml.datasets.EtherBits import EtherBits

from modules.network import Network
from trainer import Trainer

path = "../../"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainSet = EtherBits(path, train=True, frame_size=1518, smallDataset=True)
testSet = EtherBits(path, train=False, frame_size=1518, smallDataset=True)
batch = 5000
print(len(trainSet))
print(len(testSet))
trainLoader = DataLoader(trainSet, batch_size=batch, shuffle=False)
testLoader = DataLoader(testSet, batch_size=batch, shuffle=False)

model = Network()

loss_fn = nn.MSELoss(reduction='mean')

trainingManager = Trainer(model, loss_fn, 1E-3, trainLoader, testLoader, device)

while True:
    command = input("Next command: ")

    if command == 'train':
        epochs = int(input("Number of epochs: "))
        trainingManager.train(epochs)

    if command == 'test':
        trainingManager.test()

    if command == 'quit':
        break

    if command == 'save':
        path = input("Path: ")
        trainingManager.saveModel(path)

    if command == 'load':
        path = input("Path: ")
        trainingManager.loadModel(path)


    if command == 'learning_rate':
        rate = float(input("Rate: "))
        trainingManager.setLearningRate(rate)

