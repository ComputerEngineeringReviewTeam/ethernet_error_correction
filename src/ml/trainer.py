import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import F
from torch.optim.optimizer import Optimizer


class Trainer:
    """
    Simple class that trains and evaluates given Module using the given DataLoaders
    """
    def __init__(self,
                 model: nn.Module,
                 loss_fn: F,
                 learning_rate: float,
                 train_loader: DataLoader,
                 test_loader: DataLoader = None,
                 device="cpu",
                 ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

    def setLossFn(self, loss_fn):
        self.loss_fn = loss_fn

    def setLearningRate(self, learning_rate):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def setTrainLoader(self, train_loader):
        self.train_loader = train_loader

    def train(self, epochs):
        for epoch in range(epochs):
            enumerate = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                predictions = self.model(x.to(torch.float32))
                loss = self.loss_fn(predictions, y.to(torch.float32))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f'Epoch: {epoch}, Batch {enumerate}, Loss: {loss.item()}')
                enumerate += 1

    def test(self):
        sampleCount = 0
        correctCount = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.model(x.to(torch.float32))
                for pred in range(predictions.size(0)):
                    same = True
                    for bit in range(predictions.size(1)):
                        if bool(predictions[pred][bit]) != y[pred][bit]:
                            same = False
                    correctCount += same
                sampleCount += predictions.size(0)

        print(f'Model correctly fixed {correctCount} frames out of {sampleCount}')

