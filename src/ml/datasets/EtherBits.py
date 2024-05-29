from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Unpacks bytes to torch.tensor.bool"""
    return torch.from_numpy(np.unpackbits(np.frombuffer(data, dtype=np.uint8)))


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


class EtherBits_NEW(Dataset):
    """
    Base class for the EtherBits datasets.
    Implements loading Ethernet II frames from binary file and converting them to torch.tensor.bool.
    Holds and returns 2 lists of x and y tensors (usually data and labels).
    """
    def __init__(self, x_path: str, y_path: str, frame_size: int = 1518):
        self.xs = []
        self.ys = []
        with open(x_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.xs.append(_bytes_to_tensor(frame_bytes))
        with open(y_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.ys.append(_bytes_to_tensor(frame_bytes))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class EtherBitsXor(EtherBits_NEW):
    """
    EterhBits dataset containing:
        - [0] frames with errors
        - [1] xors of the frames with errors and error-free ones (i.e. 1s only on bits with errors)
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + "train/" if train else data_dir + "test/"
        x_path = dir_path + filename + ".dat"
        y_path = dir_path + filename + "_xor.dat"
        super().__init__(x_path, y_path, frame_size)


class EtherBitsOg(EtherBits_NEW):
    """
    EterhBits dataset containing:
        - [0] frames with errors
        - [1] error-free frames
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + "train/" if train else data_dir + "test/"
        x_path = dir_path + filename + ".dat"
        y_path = dir_path + filename + "_og.dat"
        super().__init__(x_path, y_path, frame_size)


class EtherBitsH_NEW(Dataset):
    """
    Base class for the homogenous EtherBits datasets.
    Implements loading Ethernet II frames from binary file and converting them to torch.tensor.bool.
    Holds and returns only 1 list of tensors but returns a tuple with TWO references
    (to follow the convention)
    """
    def __init__(self, data_path: str, frame_size: int = 1518):
        self.data = []
        with open(data_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.data.append(_bytes_to_tensor(frame_bytes))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx]


class EtherBitsHXor(EtherBitsH_NEW):
    """
    EterhBits homogenous dataset containing:
        - [0] and [1] xors of the frames with errors and error-free ones (i.e. 1s only on bits with errors)
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + "train/" if train else data_dir + "test/"
        x_path = dir_path + filename + "_xor.dat"
        super().__init__(x_path, frame_size)


class EtherBitsHOg(EtherBitsH_NEW):
    """
    EterhBits homogenous dataset containing:
        - [0] and [1] error-free frames
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + "train/" if train else data_dir + "test/"
        x_path = dir_path + filename + "_og.dat"
        super().__init__(x_path, frame_size)


class EtherBitsHErr(EtherBitsH_NEW):
    """
    EterhBits homogenous dataset containing:
        - [0] and [1] frames with errors
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + "train/" if train else data_dir + "test/"
        x_path = dir_path + filename + ".dat"
        super().__init__(x_path, frame_size)
