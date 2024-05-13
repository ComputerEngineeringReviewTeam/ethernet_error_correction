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
                 optimizer: Optimizer,
                 train_loader: DataLoader,
                 test_loader: DataLoader = None,
                 device="cpu",
                 ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def _train(self, training_report_interval):
        size = len(self.train_loader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if training_report_interval is not None:
                if batch % training_report_interval == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test(self):
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def run(self, epochs: int, training_report_interval: int | None = None, test: bool = False):
        if test and self.test_loader is None:
            print("No DataLoader for test data provided!")

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self._train(training_report_interval)
            if test and self.test_loader is not None:
                self._test()
        print("Done!")
