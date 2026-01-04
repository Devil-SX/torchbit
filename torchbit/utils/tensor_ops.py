"""
Tensor utility functions for verification.

Provides helper functions for tensor manipulation and conversion
to hex representation.
"""
import torch
from ..core.dtype import torch_to_std_dtype
from typing import Optional


def tensor2vector(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to 2D with last dimension preserved.

    Reshapes a tensor so that all dimensions except the last are
    collapsed into the first dimension, keeping the last dimension
    unchanged.

    Args:
        tensor: Input PyTorch tensor of any shape.

    Returns:
        2D tensor with shape (-1, tensor.size(-1)).

    Example:
        >>> tensor = torch.randn(3, 4, 5, 6)
        >>> result = tensor2vector(tensor)
        >>> result.shape
        torch.Size([12, 6])
    """
    tensor = tensor.view(-1, tensor.size(-1))
    return tensor


def get_hex(x: torch.Tensor, dtype: torch.dtype = None) -> str:
    """Get hex representation of a scalar tensor.

    Converts a scalar tensor to its hexadecimal representation
    using the standard integer dtype for its bit width.

    Args:
        x: Scalar tensor (single element tensor).
        dtype: Optional dtype for conversion. If None, uses x.dtype.

    Returns:
        Hex string representation (e.g., "0xDEADBEEF").

    Raises:
        AssertionError: If x is not a torch.Tensor or has more than 1 element.

    Example:
        >>> x = torch.tensor(0xDEADBEEF, dtype=torch.int32)
        >>> get_hex(x)
        '0xdeadbeef'
        >>> x = torch.tensor(3.14159, dtype=torch.float32)
        >>> get_hex(x)
        '0x40490ffd'
    """
    assert isinstance(x, torch.Tensor)
    if dtype is None:
        dtype = x.dtype
    x = x.to(dtype)
    x_int = torch_to_std_dtype(x)
    return hex(x_int.item())
