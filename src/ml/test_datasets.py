import torch
from torch.utils.data import DataLoader

from datasets.EtherBits import EtherBits
from datasets.EtherBytes import EtherBytes

test_data = "../../data/prep/capture_test.dat"
test_xors = "../../data/prep/capture_test_xor.dat"
bits = EtherBits(test_data, test_xors, frame_size=1518)
byts = EtherBytes(test_data, test_xors, frame_size=1518)

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
    print(X)
    print(y)
    break

print("\tByte Loader [0]")
for X, y in byte_loader:
    print(X)
    print(y)
    break