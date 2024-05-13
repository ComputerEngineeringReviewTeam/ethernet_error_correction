from torch.utils.data import Dataset

from .funcs import bit_tensor


class EtherBits(Dataset):
    """
    Dataset of Ethernet II frames encoded with 8b/10b encoding.
    Each frame is converted into a torch.tensor of bits

    :param storage: defines whatever frames should be stored in memory as bytes objects or already as tensors
                    the former will be slower when loading frames into network and the latter while first loading
                    the dataset into memory. "tensor" by default - frames are stored as tensors
    """
    def __init__(self, filepath: str, frame_size: int = 1518, storage: str = "tensor"):
        self.frames = []
        self.storage = storage
        with open(filepath, "rb") as f:
            while frame_bytes := f.read(frame_size):
                if self.storage == "tensor":
                    self.frames.append(bit_tensor(frame_bytes))
                else:
                    self.frames.append(frame_bytes)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if self.storage == "tensor":
            return self.frames[idx]
        else:
            return bit_tensor(self.frames[idx])
