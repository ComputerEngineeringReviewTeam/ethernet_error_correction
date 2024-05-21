import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.EtherBits import EtherBits

from network import Network

path = "../../"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainSet = EtherBits(path, train=True, frame_size=1518, smallDataset=True)
testSet = EtherBits(path, train=False, frame_size=1518, smallDataset=True)

trainLoader = DataLoader(trainSet, batch_size=5, shuffle=True)
testLoader = DataLoader(testSet, batch_size=5, shuffle=True)

model = Network()

loss_fn = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1):
    for x, y in trainLoader:
        x, y = x.to(device), y.to(device)

        predictions = model(x.to(torch.float32))
        loss = loss_fn(predictions, y.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

sampleCount = 0
correctCount = 0

with torch.no_grad():
    for x, y in testLoader:
        x, y = x.to(device), y.to(device)
        predictions = model(x.to(torch.float32))
        for pred in range(predictions.size(0)):
            same = True
            for bit in range(predictions.size(1)):
                if bool(predictions[pred][bit]) != y[pred][bit]:
                    same = False
            correctCount += same
        sampleCount += predictions.size(0)


print(correctCount)
print(sampleCount)
