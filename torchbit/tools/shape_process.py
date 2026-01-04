"""
Tensor padding and reshaping utilities for hardware interfaces.

Provides functions for:
- Calculating padding to achieve memory alignment
- Padding tensors to specific alignments
- Removing padding to restore original dimensions

Essential for ensuring tensor dimensions meet hardware interface requirements.
"""
import torch
import torch.nn.functional as F


def get_padlen(dim_size: int, align: int) -> int:
    """Calculate padding length needed to align a dimension.

    Computes how many elements need to be added to reach the next
    aligned boundary.

    Args:
        dim_size: Current dimension size.
        align: Alignment boundary (must be positive).

    Returns:
        Number of elements to add. Returns 0 if already aligned.

    Example:
        >>> get_padlen(232, 16)
        8  # Need 8 elements to reach 240 (next multiple of 16)
        >>> get_padlen(240, 16)
        0  # Already aligned
    """
    return (align - dim_size % align) % align


def pad(tensor: torch.Tensor, dim: int, align: int, pad_last: bool = True, pad_value: float = 0) -> torch.Tensor:
    """Pad a tensor dimension to achieve alignment.

    Adds padding elements (default zero) to the specified dimension
    to make its size a multiple of the alignment value.

    Args:
        tensor: Input PyTorch tensor.
        dim: Dimension index to pad (0-indexed).
        align: Target alignment boundary.
        pad_last: If True, append padding at the end of the dimension.
                  If False, prepend padding at the start.
        pad_value: Value to use for padding elements. Defaults to 0.

    Returns:
        A new tensor with the padded dimension. Shape of other
        dimensions remains unchanged.

    Raises:
        AssertionError: If dim is out of bounds.

    Example:
        >>> x = torch.randn(2, 232)  # Not aligned to 16
        >>> x_padded = pad(x, 1, 16)  # Pad dimension 1 to 240
        >>> x_padded.shape
        torch.Size([2, 240])
        >>>
        >>> # Pad at the beginning instead of end
        >>> x_prepend = pad(x, 1, 16, pad_last=False)
        >>> x_prepend.shape
        torch.Size([2, 240])
    """
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


def depad(tensor: torch.Tensor, dim: int, original_dim: int, depad_last: bool = True) -> torch.Tensor:
    """Remove padding to restore original tensor dimensions.

    Extracts a subset of elements from the specified dimension,
    removing previously added padding.

    Args:
        tensor: Padded PyTorch tensor.
        dim: Dimension index to depad (0-indexed).
        original_dim: Target size after removing padding.
        depad_last: If True, remove elements from the end.
                    If False, remove elements from the start.

    Returns:
        A new tensor with the original dimension size.

    Raises:
        AssertionError: If dim is out of bounds or original_dim is invalid.

    Example:
        >>> x = torch.randn(2, 240)  # Padded from original 232
        >>> x_restored = depad(x, 1, 232)
        >>> x_restored.shape
        torch.Size([2, 232])
        >>>
        >>> # If padding was at the beginning
        >>> x_prepend = torch.randn(2, 240)
        >>> x_deprepend = depad(x_prepend, 1, 232, depad_last=False)
        >>> x_deprepend.shape
        torch.Size([2, 232])
    """
    shape = tensor.shape
    assert dim < len(shape)

    depad_shape = list(shape)
    depad_shape[dim] = original_dim

    # advanced indexing to create new tensor
    slices = [slice(None)] * len(shape)
    if depad_last:
        slices[dim] = slice(0, original_dim)
    else:
        padlen = tensor.shape[dim] - original_dim
        slices[dim] = slice(padlen, None)
    return tensor[tuple(slices)].clone()


if __name__ == '__main__':
    ORIG = 232
    ALIGN = 16
    x = torch.randn(2, ORIG)

    print(f"original shape is {x.shape}")
    x_pad = pad(x, 1, ALIGN)
    print(f"padded shape is {x_pad.shape}")
    x_depad = depad(x_pad, 1, ORIG)
    print(f"depadded shape is {x_depad.shape}")
