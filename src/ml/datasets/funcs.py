import torch
from functools import reduce


def bitify(byte: int):
    return [(byte << i) & 1 for i in range(7, -1, -1)]


def bit_tensor(frame: bytes):
    bit_stream = reduce(lambda a, b: a + b, list(map(bitify, frame)))
    return torch.tensor(bit_stream, dtype=torch.bool)


def byte_tensor(frame: bytes):
    return torch.tensor(*list(frame), dtype=torch.uint8)
