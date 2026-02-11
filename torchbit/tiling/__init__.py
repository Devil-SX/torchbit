"""
Tensor tiling, reshaping, and memory mapping utilities.

This module provides functionality for:
- Padding tensors to meet hardware alignment requirements
- Mapping tensors to memory layouts with specific tiling patterns
- Converting between tensor and hardware interface formats
"""

from .pad_utils import get_padlen, pad, depad, depad_like
from .address_mapping import AddressMapping, ContiguousAddressMapping
from .tile_mapping import (
    TileMapping,
    matrix_to_logic_seq,
    logic_seq_to_matrix,
    array_to_logic_seq,
    logic_seq_to_array,
)

__all__ = [
    # pad_utils
    "get_padlen",
    "pad",
    "depad",
    "depad_like",
    # address_mapping
    "AddressMapping",
    "ContiguousAddressMapping",
    # tile_mapping
    "TileMapping",
    "matrix_to_logic_seq",
    "logic_seq_to_matrix",
    "array_to_logic_seq",
    "logic_seq_to_array",
]
