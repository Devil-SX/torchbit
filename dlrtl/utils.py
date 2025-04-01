import torch


def compress(tensor: torch.Tensor):
    tensor = tensor.view(-1, tensor.size(-1))
    return tensor

def get_bit(value:int, num:int):
    assert num >= 0
    return (value * 2 **num) % 2