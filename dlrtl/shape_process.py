import torch
import torch.nn.functional as F

# pad: original dim + padlen -> paddim
# depad: paddim - padlen -> original dim

def get_padlen(dim_size, align):
    return (align - dim_size % align) % align

def pad(tensor, dim:int, align:int):
    shape = tensor.shape
    assert dim < len(shape)
    dtype = tensor.dtype

    zero_shape = list(shape)
    zero_shape[dim] = get_padlen(shape[dim], align)

    zeros = torch.zeros(zero_shape, dtype=dtype)
    return torch.cat([tensor, zeros], dim=dim).clone()

def depad(tensor, dim:int, orignal_dim:int):
    shape = tensor.shape
    assert dim < len(shape)

    depad_shape = list(shape)
    depad_shape[dim] = orignal_dim

    # advanced indexing to create new tensor
    slices = [slice(None)] * len(shape)
    slices[dim] = slice(0, orignal_dim)
    return tensor[tuple(slices)].clone()


if __name__ == '__main__':
    ORIG = 232
    ALIGN = 16
    x = torch.randn(2,ORIG)

    print(f"original shape is {x.shape}")
    x_pad = pad(x, 1, ALIGN)
    print(f"padded shape is {x_pad.shape}")
    x_depad = depad(x_pad, 1, ORIG)
    print(f"depadded shape is {x_depad.shape}")
