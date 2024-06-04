import numpy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.etherbits import *

from src.ml.modules.network import Network
from src.ml.modules.autoencoder import Autoencoder

path = "../../data/prep/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# trainSet = EtherBits(path, train=True, frame_size=1518, smallDataset=True)
# testSet = EtherBits(path, train=False, frame_size=1518, smallDataset=True)
trainSet = EtherBitsXor(path, "big", train=True, frame_size=4)
testSet = EtherBitsXor(path, "big", train=False, frame_size=4)
batch = 1000
print(len(trainSet))
print(len(testSet))
trainLoader = DataLoader(trainSet, batch_size=batch, shuffle=True)
testLoader = DataLoader(testSet, batch_size=batch, shuffle=True)

model = Network().to(device=device)

loss_fn = nn.modules.loss.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for epoch in range(20):
    for i, (x, y) in enumerate(trainLoader):
        x, y = x.to(device), y.to(device)

        predictions = model(x.to(torch.float32))
        loss = loss_fn(predictions, y.to(torch.float32))
        # loss = loss_fn((predictions * 2).to(torch.uint8).to(torch.bool).to(torch.float32), y.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch} - batch: {i}, Loss: {loss.item()}')
print("\nTraining finished")

test_loss, correct = 0, 0
all = 0
bits = 0
one_err = 0
all_err = 0
model.eval()
with torch.no_grad():
    for x, y in testLoader:
        # if bits == 0:
        #     print(x[:, 96:111])
        #     print(x.shape)
        # if torch.count_nonzero(x[:, 96:111] ^ y[:, 96:111]).item() != 0:
        #     continue

        x, y = x.to(device), y.to(device)
        pred = model(x.to(torch.float32))

        yf = y.to(torch.float32)
        pred_b = (pred * 2).to(torch.uint8).to(torch.bool)
        diff = torch.count_nonzero(pred_b ^ y, dim=1)
        bits += diff.sum().item()
        test_loss += loss_fn(pred, yf).item()
        correct += diff.shape[0] - torch.count_nonzero(diff, dim=0)
        one_err += diff.shape[0] - torch.count_nonzero(diff - 1.0, dim=0)
        all += pred.shape[0]
        all_err += torch.count_nonzero(x ^ y)


print(test_loss / len(testLoader))
print(bits / (len(testLoader) * batch))
print(correct.item(), "/", all, "=", correct.item() / all)
print(one_err.item(), "/", all, "=", one_err.item() / all)
print(all_err.item() / all)

