import torch
from torch.utils.data import Dataset


class EtherBytes(Dataset):
    """
    Dataset of Ethernet II frames encoded with 8b/10b encoding.
    Each frame is converted into a torch.tensor of bytes
    """
    def __init__(self, directory: str, train: bool, frame_size=1518, smallDataset: bool = False):
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
                self.frames.append(torch.frombuffer(frame_bytes, dtype=torch.uint8))
        with open(xorpath, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.xors.append(torch.frombuffer(frame_bytes, dtype=torch.uint8))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.xors[idx]
