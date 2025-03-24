import torch


def compress(tensor: torch.Tensor):
    tensor = tensor.view(-1, tensor.size(-1))
    return tensor