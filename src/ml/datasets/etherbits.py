"""
Various datasets of Ethernet II frames, expressed as tensors of singular bits (bool), extending torch.utils.data.Dataset

List:
- EtherBits line:
  - EtherBitsXor - dataset containing frames with errors in them and xors of them and their error-free versions
  - EtherBitsOg -  dataset containing frames with errors in them and their error-free versions
- EtherBitsH line - input and output data are the same:
  - EtherBitsH- dataset containing frames with errors in them
  - EtherBitsOg- dataset containing error-free frames
  - EtherBitsXor - dataset containing frames with errors in them XORed with their error-free versions

Author: Marek Szymański
"""

import torch
from typing import Tuple
from torch.utils.data import Dataset

from src.ml.util.convert_tensors import bytes_to_bool_tensor


DATA_FILE_EXTENSION = '.dat'
DESC_FILE_EXTENSION = '.csv'

CLEAN_FRAMES_TAG = "_og"
ERROR_FRAMES_TAG = ""
XOR_FRAMES_TAG = "_xor"
DESC_FILE_TAG = "_errDesc"

TRAIN_DIR = "train/"
TEST_DIR = "test/"


class EtherBits(Dataset):
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
                self.xs.append(bytes_to_bool_tensor(frame_bytes))
        with open(y_path, "rb") as f:
            while frame_bytes := f.read(frame_size):
                self.ys.append(bytes_to_bool_tensor(frame_bytes))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


class EtherBitsXor(EtherBits):
    """
    EtherBits dataset containing:
        - [0] frames with errors
        - [1] xors of the frames with errors and error-free ones (i.e. 1s only on bits with errors)
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + ERROR_FRAMES_TAG + DATA_FILE_EXTENSION
        y_path = dir_path + filename + XOR_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, y_path, frame_size)


class EtherBitsOg(EtherBits):
    """
    EtherBits dataset containing:
        - [0] frames with errors
        - [1] error-free frames
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + ERROR_FRAMES_TAG + DATA_FILE_EXTENSION
        y_path = dir_path + filename + CLEAN_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, y_path, frame_size)


class EtherBitsH(Dataset):
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
                self.data.append(bytes_to_bool_tensor(frame_bytes))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx]


class EtherBitsHXor(EtherBitsH):
    """
    EtherBits homogenous dataset containing:
        - [0] and [1] xors of the frames with errors and error-free ones (i.e. 1s only on bits with errors)
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + XOR_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, frame_size)


class EtherBitsHOg(EtherBitsH):
    """
    EtherBits homogenous dataset containing:
        - [0] and [1] error-free frames
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + CLEAN_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, frame_size)


class EtherBitsHErr(EtherBitsH):
    """
    EtherBits homogenous dataset containing:
        - [0] and [1] frames with errors
    """
    def __init__(self, data_dir: str, filename: str, frame_size: int = 1518, train: bool = True):
        dir_path = data_dir + TRAIN_DIR if train else data_dir + TEST_DIR
        x_path = dir_path + filename + ERROR_FRAMES_TAG + DATA_FILE_EXTENSION
        super().__init__(x_path, frame_size)
