import os

from torch.utils.data import Dataset

from .funcs import bit_tensor


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
            filepath = directory + 'capture.dat'
            xorpath = directory + 'capture_xor.dat'
        self.frames = []
        self.xors = []
        with open(filepath, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.frames.append(bit_tensor(frame_bytes))
        with open(xorpath, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.xors.append(bit_tensor(frame_bytes))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.xors[idx]
