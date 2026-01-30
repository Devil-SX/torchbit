"""
Tensor tiling, reshaping, and memory mapping utilities.

This module provides functionality for:
- Padding tensors to meet hardware alignment requirements
- Mapping tensors to memory layouts with specific tiling patterns
- Converting between tensor and hardware interface formats
"""

from .pad_utils import get_padlen, pad, depad, depad_like
from .mapping import AddressMapping, TileMapping, tensor_to_cocotb_seq, cocotb_seq_to_tensor

__all__ = [
    # pad_utils
    "get_padlen",
    "pad",
    "depad",
    "depad_like",
    # mapping
    "AddressMapping",
    "TileMapping",
    "tensor_to_cocotb_seq",
    "cocotb_seq_to_tensor",
]
