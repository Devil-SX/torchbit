import torch
import torch.nn.functional as F

# pad: original dim + padlen -> paddim
# depad: paddim - padlen -> original dim

def get_padlen(dim_size, align):
    return (align - dim_size % align) % align

def pad(tensor, dim:int, align:int, pad_last=True, pad_value=0):
    shape = tensor.shape
    assert dim < len(shape)
    dtype = tensor.dtype

    zero_shape = list(shape)
    zero_shape[dim] = get_padlen(shape[dim], align)

    zeros = torch.ones(zero_shape, dtype=dtype) * pad_value
    if pad_last:
        return torch.cat([tensor, zeros], dim=dim).clone()
    else:
        return torch.cat([zeros, tensor], dim=dim).clone()


def depad(tensor, dim:int, orignal_dim:int, depad_last=True):
    shape = tensor.shape
    assert dim < len(shape)

    depad_shape = list(shape)
    depad_shape[dim] = orignal_dim

    # advanced indexing to create new tensor
    slices = [slice(None)] * len(shape)
    if depad_last:
        slices[dim] = slice(0, orignal_dim)
    else:
        padlen = tensor.shape[dim] - orignal_dim
        slices[dim] = slice(padlen, None)
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
