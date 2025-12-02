import torch
from ..core.dtype import torch_to_std_dtype


def tensor2vector(tensor: torch.Tensor):
    tensor = tensor.view(-1, tensor.size(-1))
    return tensor

def get_hex(x: torch.Tensor, dtype: torch.dtype=None):
    assert isinstance(x, torch.Tensor)
    if dtype is None:
        dtype = x.dtype
    x = x.to(dtype)
    x_int = torch_to_std_dtype(x)
    return hex(x_int.item())

