"""
Simple Trainer class that trains and evaluates given Module using the given DataLoaders

Author: Mateusz Nurczyński
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import F
from torch.optim.optimizer import Optimizer
import util.convert_tensors
from src.ml.util import convert_tensors


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
                 optimizer_generator = lambda learning_rate, model: torch.optim.Adam(model.parameters(), lr=learning_rate),
                 ):
        self.optimizer_generator = optimizer_generator
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.setLearningRate(learning_rate)
        self.device = device

    def setLossFn(self, loss_fn):
        self.loss_fn = loss_fn

    def setLearningRate(self, learning_rate):
        self.optimizer = self.optimizer_generator(learning_rate, self.model)

    def setTrainLoader(self, train_loader):
        self.train_loader = train_loader

    def setTestLoader(self, test_loader):
        self.test_loader = test_loader

    def saveModel(self, model_path):
        torch.save(self.model.state_dict(), "models/"+model_path+".model")

    def loadModel(self, model_path):
        self.model.load_state_dict(torch.load("models/"+model_path+".model"))

    def train(self, epochs, reportEveryXBatches=1):
        self.model.train()
        for epoch in range(epochs):
            batchNumber = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                predictions = self.model(x.to(torch.float32))
                loss = self.loss_fn(predictions, y.to(torch.float32))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batchNumber % reportEveryXBatches == 0:
                    print(f'Epoch: {epoch}, Batch {batchNumber}, Loss: {loss.item()}')
                batchNumber += 1

    def test(self):
        self.model.eval()
        sampleCount = 0
        correctCount = 0
        frameCount = 0
        correctFrameCount = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.model(x.to(torch.float32))
                predictions = convert_tensors.round_float_to_bool_tensor(predictions)
                sampleCount += predictions.size(0)*predictions.size(1)
                comp = predictions == y
                correctCount += torch.count_nonzero(comp)
                comp = torch.logical_not(comp)
                frameCount += predictions.size(0)
                comp = torch.sum(comp, 1)
                correctFrameCount += predictions.size(0)-torch.count_nonzero(comp)


        print(f'Model correctly predicted {correctCount} bits out of {sampleCount}')
        print(f'Model correctly predicted {correctFrameCount} frames out of {frameCount}')
