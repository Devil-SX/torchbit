"""
Type mappings and bit-width definitions for hardware verification.

This module provides mappings between PyTorch/numpy dtypes and their bit widths,
which are essential for:
- Converting tensors to packed integer representations
- Reading/writing memory files with correct element sizes
- Understanding data layout in HDL interfaces

Supported dtypes:
- PyTorch: uint8, int8, int16, int32, int64, float16, bfloat16, float32, float64
- Numpy: int8, int16, int32, int64 (with dtype objects)

Key dictionaries:
- dtype_to_bits: Maps PyTorch dtype to bit width
- standard_torch_dtype: Maps bit width to standard PyTorch int dtype
- standard_numpy_dtype: Maps bit width to standard numpy dtype
- standard_be_numpy_dtype: Big-endian numpy dtypes by bit width
- standard_le_numpy_dtype: Little-endian numpy dtypes by bit width
"""
import torch
import numpy as np

dtype_to_bits: dict[torch.dtype, int] = {
    # for unsigned int, only support uint8. uint16 and uint32 are limited support
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
    torch.float64: 64,
}

standard_torch_dtype: dict[int, torch.dtype] = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}

numpy_dtype_to_bits: dict[np.dtype, int] = {
    np.int8: 8,
    np.int16: 16,
    np.int32: 32,
    np.int64: 64,
    np.dtype("int8"): 8,
    np.dtype("int16"): 16,
    np.dtype("int32"): 32,
    np.dtype("int64"): 64,
}

standard_numpy_dtype: dict[int, np.dtype] = {
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
}
standard_be_numpy_dtype: dict[int, np.dtype] = {
    8: np.dtype(">i1"),
    16: np.dtype(">i2"),
    32: np.dtype(">i4"),
    64: np.dtype(">i8"),
}
standard_le_numpy_dtype: dict[int, np.dtype] = {
    8: np.dtype("<i1"),
    16: np.dtype("<i2"),
    32: np.dtype("<i4"),
    64: np.dtype("<i8"),
}


def torch_to_std_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to its standard integer dtype representation.

    Converts floating-point or non-standard dtypes to the standard integer
    dtype for the corresponding bit width. This is useful for bit-level
    operations and memory file I/O.

    Args:
        tensor: Input PyTorch tensor of any supported dtype.

    Returns:
        A view of the tensor with the standard integer dtype for its
        bit width (int8 for 8-bit, int16 for 16-bit, etc.).

    Raises:
        KeyError: If the tensor's dtype is not in dtype_to_bits.

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1.5, 2.5], dtype=torch.float32)
        >>> std_tensor = torch_to_std_dtype(tensor)
        >>> std_tensor.dtype
        torch.int32
    """
    bits = dtype_to_bits[tensor.dtype]
    std_t_d = standard_torch_dtype[bits]
    return tensor.view(std_t_d)
