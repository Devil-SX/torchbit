"""
Tensor-to-memory tiling and rearrangement utilities.

Provides TileMapping for configuring how software tensors should be
rearranged into hardware vector sequences (also known as hardware matrices)
for storage in memory arrays. Uses einops-style dimension notation.

A hardware vector sequence is a 2D layout where the first dimension
(temporal) is ordered by time (lower index = earlier time step), and the
second dimension (spatial) represents the hardware vector / parallelism.
"""
import torch
import numpy as np
import einops
from ..core.vector import Vector
from ..core.int_sequence import IntSequence
from dataclasses import dataclass


def _is_2d_hw_format(pattern: str) -> bool:
    """Check if pattern is a valid 2D hardware vector sequence format.

    Valid formats:
    - "(temporal_dims) (spatial_dims)" - both groups parenthesized
    - "temporal (spatial_dims)" - single temporal dim + parenthesized spatial
    - "(temporal_dims) spatial" - parenthesized temporal + single spatial

    Args:
        pattern: Einops pattern string to validate.

    Returns:
        True if pattern is valid 2D format, False otherwise.
    """
    # Remove extra spaces
    pattern = pattern.strip()

    # Count top-level groups (parenthesized) and standalone dimensions
    groups = []
    depth = 0
    current_group = ""
    standalone_dims = []

    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == '(':
            depth += 1
            if depth == 1:
                current_group = ""
            i += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                groups.append(current_group.strip())
            i += 1
        elif depth == 0:
            # Not in parentheses, collect standalone dimension name
            dim_name = ""
            while i < len(pattern) and pattern[i] not in (' ', ')'):
                dim_name += pattern[i]
                i += 1
            if dim_name:
                standalone_dims.append(dim_name.strip())
            # Skip whitespace
            while i < len(pattern) and pattern[i] == ' ':
                i += 1
        elif depth == 1:
            current_group += char
            i += 1
        else:
            i += 1

    # Total groups = parenthesized groups + standalone dimensions
    total_groups = len(groups) + len(standalone_dims)
    return total_groups == 2


@dataclass
class TileMapping:
    """Defines tensor-to-memory tiling and rearrangement.

    Configures how a software tensor should be rearranged into a hardware
    vector sequence (also known as hardware matrix) for storage in a memory
    array. Uses einops-style dimension notation to specify the transformation
    between software (tensor) layout and hardware (2D) layout.

    The resulting 2D layout is a hardware vector sequence where:
    - First dimension (temporal): ordered by time, lower index = earlier
      time step (index 0 is the first time step, index 1 is the second, etc.)
    - Second dimension (spatial): the hardware vector, reflecting parallelism

    TileMapping handles only value conversion (tensor <-> IntSequence).
    Address generation is handled separately by AddressMapping.

    Key concepts:
    - sw_einops: Software tensor dimension layout (e.g., "(ht hs) (wt ws) ct")
    - hw_einops: Hardware vector sequence layout (e.g., "(ht wt ct) (hs ws)")
    - hw_temp_dim: Temporal dimension sizes (first group of hw_einops)
    - hw_spat_dim: Spatial dimension sizes (second group of hw_einops)

    Attributes:
        dtype (torch.dtype): Data type of tensor elements.
        sw_einops (str): Software dimension layout without "->".
        hw_einops (str): Hardware vector sequence layout "(temporal) (spatial)".
        hw_temp_dim (dict): Hardware temporal dimension sizes.
        hw_spat_dim (dict): Hardware spatial dimension sizes.

    Example:
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="(ht hs) (wt ws) ct",
        ...     hw_einops="(ht wt ct) (hs ws)",
        ...     hw_temp_dim={"ht": 2, "wt": 2, "ct": 3},
        ...     hw_spat_dim={"hs": 4, "ws": 4},
        ... )

    Simple example:
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="c h w",
        ...     hw_einops="c (h w)",
        ...     hw_temp_dim={"c": 3},
        ...     hw_spat_dim={"h": 8, "w": 8},
        ... )
        >>> tensor = torch.randn(3, 8, 8)
        >>> values = mapping.to_hw(tensor)
        >>> tensor_restored = mapping.to_sw(values)
    """

    dtype: torch.dtype
    sw_einops: str
    hw_einops: str
    hw_temp_dim: dict
    hw_spat_dim: dict

    def __post_init__(self):
        """Validate and compute derived attributes."""
        assert all(isinstance(k, str) for k in self.hw_temp_dim.keys()) and all(
            isinstance(v, int) for v in self.hw_temp_dim.values()
        ), "hw_temp_dim must be a dict of str to int"
        assert all(isinstance(k, str) for k in self.hw_spat_dim.keys()) and all(
            isinstance(v, int) for v in self.hw_spat_dim.values()
        ), "hw_spat_dim must be a dict of str to int"

        # Validate hw_einops is 2D format: (temporal_dims) (spatial_dims)
        assert _is_2d_hw_format(self.hw_einops), \
            f"hw_einops must be 2D format '(temporal) (spatial)', got: {self.hw_einops}"

        # Number of elements per temporal unit (spatial size)
        self.num = int(np.prod(list(self.hw_spat_dim.values())))

        # Build einops formulas: sw <-> hw transformation
        # sw_to_hw: sw_einops -> hw_einops
        # hw_to_sw: hw_einops -> sw_einops
        self.sw_to_hw_formula = f"{self.sw_einops} -> {self.hw_einops}"
        self.hw_to_sw_formula = f"{self.hw_einops} -> {self.sw_einops}"

    def to_hw(self, tensor: torch.Tensor) -> IntSequence:
        """Convert a software tensor to hardware vector sequence (values only).

        Transforms the tensor according to the TileMapping and produces
        an IntSequence of packed integer values, ordered by time step
        (index 0 = earliest time step).

        Args:
            tensor: Input PyTorch tensor with shape matching sw_einops pattern.

        Returns:
            IntSequence of packed integers, one per temporal unit,
            ordered from earliest to latest time step.

        Example:
            >>> mapping = TileMapping(
            ...     dtype=torch.float32,
            ...     sw_einops="c h w",
            ...     hw_einops="c (h w)",
            ...     hw_temp_dim={"c": 3},
            ...     hw_spat_dim={"h": 32, "w": 32},
            ... )
            >>> tensor = torch.randn(3, 32, 32)
            >>> values = mapping.to_hw(tensor)
            >>> # len(values) == 3
        """
        # Rearrange tensor from software layout to hardware vector sequence layout
        tensor_seq = einops.rearrange(
            tensor,
            self.sw_to_hw_formula,
            **self.hw_temp_dim,
            **self.hw_spat_dim,
        )  # [N, M] where N is temporal iterations, M is spatial elements

        # Convert each temporal row to packed integer
        values_list = IntSequence(
            Vector.from_tensor(tensor_row).to_cocotb() for tensor_row in tensor_seq
        )

        return values_list

    def to_sw(self, values: IntSequence) -> torch.Tensor:
        """Convert hardware vector sequence back to software tensor (values only).

        Takes time-ordered values from hardware and reconstructs the original
        tensor layout according to the TileMapping.

        Args:
            values: IntSequence of packed integers from hardware.

        Returns:
            A PyTorch tensor with shape matching sw_einops pattern.

        Example:
            >>> values = mapping.to_hw(tensor)
            >>> tensor_restored = mapping.to_sw(values)
            >>> assert torch.allclose(tensor, tensor_restored)
        """
        # Convert each integer back to tensor row
        tensor_seq = [
            Vector.from_cocotb(int_value, self.num, self.dtype).to_tensor()
            for int_value in values
        ]

        # Stack and rearrange from hardware layout to software layout
        tensor = einops.rearrange(
            tensor_seq,
            self.hw_to_sw_formula,
            **self.hw_temp_dim,
            **self.hw_spat_dim,
        )

        return tensor


def tensor_to_cocotb_seq(tensor: torch.Tensor, mapping: TileMapping) -> IntSequence:
    """Convert a tensor to a time-ordered sequence of Cocotb-compatible integer values.

    Uses the TileMapping to rearrange the tensor into a hardware vector
    sequence and pack each row into an integer suitable for HDL interfaces.
    The resulting sequence is ordered by time step (index 0 = earliest).

    Args:
        tensor: Input PyTorch tensor.
        mapping: TileMapping defining the tensor-to-memory transformation.

    Returns:
        IntSequence of integers, one per temporal unit, ready for HDL assignment.

    Example:
        >>> tensor = torch.randn(3, 32, 32)
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="c h w",
        ...     hw_einops="c (h w)",
        ...     hw_temp_dim={"c": 3},
        ...     hw_spat_dim={"h": 32, "w": 32},
        ... )
        >>> seq = tensor_to_cocotb_seq(tensor, mapping)
    """
    return mapping.to_hw(tensor)


def cocotb_seq_to_tensor(cocotb_seq: IntSequence, mapping: TileMapping) -> torch.Tensor:
    """Convert a sequence of Cocotb values back to a tensor.

    Unpacks each integer in the sequence and uses the TileMapping
    to rearrange into the original tensor layout.

    Args:
        cocotb_seq: IntSequence of integers from HDL interface.
        mapping: TileMapping defining the memory-to-tensor transformation.

    Returns:
        PyTorch tensor with the original layout.

    Example:
        >>> result = cocotb_seq_to_tensor(seq, mapping)
        >>> # result.shape matches original tensor shape
    """
    return mapping.to_sw(cocotb_seq)
