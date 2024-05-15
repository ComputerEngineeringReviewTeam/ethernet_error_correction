import torch
from torch.utils.data import DataLoader

from datasets.EtherBits import EtherBits
from datasets.EtherBytes import EtherBytes

test_data = "../../data/datasets/test.dat"
bits = EtherBits(test_data, frame_size=2000)
byts = EtherBytes(test_data, frame_size=2000)

print("\tBits - frame[0]")
print(bits[0][:8], end=" ")
print(bits[0][8:16], end=" ")
print(bits[0][16:24])

print("\tBytes - frame[0]")
print(byts[0])

size = 10
bit_loader = DataLoader(bits, batch_size=size, shuffle=True)
byte_loader = DataLoader(byts, batch_size=size, shuffle=True)

print("\tBits Loader [0]")
for X in bit_loader:
    print(X)
    break

print("\tByte Loader [0]")
for X in byte_loader:
    print(X)
    break