import torch
import numpy as np

def bytes_to_bool_tensor(data: bytes) -> torch.Tensor:
    """Unpacks bytes to torch.tensor.bool"""
    return torch.from_numpy(np.unpackbits(np.frombuffer(data, dtype=np.uint8)))

def round_float_to_bool_tensor(data: torch.Tensor) -> torch.Tensor:
    """Converts float tensor to bool tensor"""
    return (data * 2).to(torch.uint8).to(torch.bool)
