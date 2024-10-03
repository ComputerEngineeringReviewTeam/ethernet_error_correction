"""
Various datasets of Ethernet II frames, expressed as tensors of bytes (uint8), extending torch.utils.data.Dataset

List:
- EtherBytes line:
  - EtherBytesXor - dataset containing frames with errors in them and xors of them and their error-free versions
  - EtherBytesOg -  dataset containing frames with errors in them and their error-free versions
- EtherBytesH line - input and output data are the same:
  - EtherBytesH- dataset containing frames with errors in them
  - EtherBytesOg- dataset containing error-free frames
  - EtherBytesXor - dataset containing frames with errors in them XORed with their error-free versions

Author: Marek Szymański
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset


DATA_FILE_EXTENSION = '.dat'
DESC_FILE_EXTENSION = '.csv'

CLEAN_FRAMES_TAG = "_og"
ERROR_FRAMES_TAG = ""
XOR_FRAMES_TAG = "_xor"
DESC_FILE_TAG = "_errDesc"

TRAIN_DIR = "train/"
TEST_DIR = "test/"


class EtherBytes(Dataset):
    """
    Base class for the EtherBytes datasets.
    Implements loading Ethernet II frames from binary file and converting them to torch.tensor.bool.
    Holds and returns 2 lists of x and y tensors (usually data and labels).
    """
    def __init__(self, x_path: str, y_path: str, frame_size: int = 1518):
        self.xs = []
        self.ys = []
        with open(x_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.xs.append(torch.frombuffer(frame_bytes, dtype=torch.uint8))
        with open(y_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.ys.append(torch.frombuffer(frame_bytes, dtype=torch.uint8))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class EtherBytesXor(EtherBytes):
    """
    EtherBytes dataset containing:
        - [0] frames with errors
        - [1] xors of the frames with errors and error-free ones (i.e. 1s only on bits with errors)
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + ERROR_FRAMES_TAG + DATA_FILE_EXTENSION
        y_path = dir_path + filename + XOR_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, y_path, frame_size)


class EtherBytesOg(EtherBytes):
    """
    EtherBytes dataset containing:
        - [0] frames with errors
        - [1] error-free frames
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + ERROR_FRAMES_TAG + DATA_FILE_EXTENSION
        y_path = dir_path + filename + CLEAN_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, y_path, frame_size)


class EtherBytesH(Dataset):
    """
    Base class for the homogenous EtherBytes datasets.
    Implements loading Ethernet II frames from binary file and converting them to torch.tensor.bool.
    Holds and returns only 1 list of tensors but returns a tuple with TWO references
    (to follow the convention)
    """
    def __init__(self, data_path: str, frame_size: int = 1518):
        self.data = []
        with open(data_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.data.append(torch.frombuffer(frame_bytes, dtype=torch.uint8))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx]


class EtherBytesHXor(EtherBytesH):
    """
    EtherBytes homogenous dataset containing:
        - [0] and [1] xors of the frames with errors and error-free ones (i.e. 1s only on bits with errors)
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + XOR_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, frame_size)


class EtherBytesHOg(EtherBytesH):
    """
    EtherBytes homogenous dataset containing:
        - [0] and [1] error-free frames
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + CLEAN_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, frame_size)


class EtherBytesHErr(EtherBytesH):
    """
    EtherBytes homogenous dataset containing:
        - [0] and [1] frames with errors
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + ERROR_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, frame_size)
