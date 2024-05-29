import torch
from torch.utils.data import DataLoader

from src.ml.datasets.EtherBits import EtherBits
from datasets.EtherBytes import EtherBytes

path = "../../"

bits = EtherBits_NEW(path, train=True, frame_size=1518, smallDataset=True)
byts = EtherBytes(path, train=False, frame_size=1518, smallDataset=True)

print("\tBits - frame[0]")
print(bits[0][0][:8], end=" ")
print(bits[0][0][8:16], end=" ")
print(bits[0][0][16:24])

print("\tBytes - frame[0]")
print(byts[0][0])

size = 10
bit_loader = DataLoader(bits, batch_size=size, shuffle=True)
byte_loader = DataLoader(byts, batch_size=size, shuffle=True)

print("\tBits Loader [0]")
for X, y in bit_loader:
    print(X.shape, y.shape)
    print(X)
    print(y)
    break

print("\tByte Loader [0]")
for X, y in byte_loader:
    print(X.shape, y.shape)
    print(X)
    print(y)
    break