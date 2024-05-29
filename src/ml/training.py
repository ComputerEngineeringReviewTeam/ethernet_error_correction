import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml.datasets.EtherBits import EtherBits

from src.ml.modules.network import Network

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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for epoch in range(2):
    for i, (x, y) in enumerate(trainLoader):
        x, y = x.to(device), y.to(device)

        predictions = model(x.to(torch.float32))
        loss = loss_fn(predictions, y.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch} - batch: {i}, Loss: {loss.item()}')
print("Training finished")

test_loss, correct = 0, 0
bits = 0
model.eval()
with torch.no_grad():
    for x, y in testLoader:
        x, y = x.to(device), y.to(device)
        pred = model(x.to(torch.float32))
        yf = y.to(torch.float32)
        diff = torch.count_nonzero((pred * 2).to(torch.uint8).to(torch.bool) ^ y, dim=0)
        bits += torch.sum(diff).item()
        test_loss += loss_fn(pred, yf).item()
        correct += (pred.equal(yf))



print(correct / len(testLoader.dataset))
print(test_loss / len(testLoader))
print(bits / (len(testLoader)  * 1518 * 8))
