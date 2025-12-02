import torch
from ..core.dtype import *


def compress(tensor: torch.Tensor):
    tensor = tensor.view(-1, tensor.size(-1))
    return tensor

def get_bit(value:int, num:int):
    assert num >= 0
    return (value * 2 **num) % 2


def get_bit_slice(value: int, high: int, low: int):
    return (value >> low) & ((1 << (high - low + 1)) - 1)

def get_hex(x: torch.Tensor, dtype: torch.dtype=None):
    assert isinstance(x, torch.Tensor)
    if dtype is None:
        dtype = x.dtype
    x = x.to(dtype)
    x_int = torch_to_std_dtype(x)
    return hex(x_int.item())