import numpy as np
import torch
from torch.utils.data import Dataset


class EtherBits(Dataset):
    """
    Dataset of Ethernet II frames encoded with 8b/10b encoding.
    Each frame is converted into a torch.tensor of bits
    """
    def __init__(self, directory: str, train: bool, frame_size: int = 1518, smallDataset: bool = False):
        if train:
            directory = directory + 'data/prep/train/'
        else:
            directory = directory + 'data/prep/test/'
        if smallDataset:
            filepath = directory + 'capture_test.dat'
            xorpath = directory + 'capture_test_xor.dat'
        else:
            filepath = directory + 'big.dat'
            xorpath = directory + 'big_xor.dat'
        self.frames = []
        self.xors = []
        with open(filepath, "rb") as f:
            while frame_bytes := f.read(frame_size):
                ndarray = np.frombuffer(frame_bytes, dtype=np.uint8)
                self.frames.append(torch.from_numpy(np.unpackbits(ndarray)))
        with open(xorpath, "rb") as f:
            while frame_bytes := f.read(frame_size):
                ndarray = np.frombuffer(frame_bytes, dtype=np.uint8)
                self.xors.append(torch.from_numpy(np.unpackbits(ndarray)))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.xors[idx]
