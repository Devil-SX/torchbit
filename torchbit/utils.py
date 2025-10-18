import torch


def compress(tensor: torch.Tensor):
    tensor = tensor.view(-1, tensor.size(-1))
    return tensor

def get_bit(value:int, num:int):
    assert num >= 0
    return (value * 2 **num) % 2


def get_bit_slice(value: int, high: int, low: int):
    return (value >> low) & ((1 << (high - low + 1)) - 1)