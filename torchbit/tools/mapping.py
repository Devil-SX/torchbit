"""
Tensor-to-memory mapping and address translation utilities.

Provides TileMapping and AddressMapping classes for configuring how tensors
should be arranged in memory. Essential for deep learning accelerator
verification where tensor data must be tiled, strided, and mapped to
specific memory layouts.
"""
from pathlib import Path
from os import PathLike
import torch
import numpy as np
import einops
from ..core.vector import Vector
from ..debug.judge import compare
from dataclasses import dataclass


class AddressMapping:
    """Maps multi-dimensional indices to flat memory addresses.

    Computes linear addresses from multi-dimensional coordinates using
    stride calculations. This is useful for:
    - Row-major or column-major tensor storage
    - Strided memory layouts
    - Multi-bank memory partitioning

    Attributes:
        strides (tuple): Stride for each dimension.
        max_values (tuple): Maximum value (exclusive) for each dimension.
        base (int): Base address offset.

    Example:
        >>> # Row-major 2D array (like C)
        >>> mapping = AddressMapping(
        ...     base=0x1000,
        ...     strides=(16, 1),  # stride of 16 for rows, 1 for cols
        ...     max_values=(4, 16)  # 4 rows, 16 columns
        ... )
        >>> addrs = mapping.get_addr_list()
        >>> # Returns addresses for all (row, col) combinations
        >>>
        >>> # Column-major 2D array (like Fortran)
        >>> col_mapping = AddressMapping(
        ...     base=0,
        ...     strides=(1, 4),  # stride of 1 for cols, 4 for rows
        ...     max_values=(16, 4)
        ... )
    """

    def __init__(self, base: int, strides: tuple, max_values: tuple):
        """Initialize AddressMapping.

        Args:
            base: Base address offset added to all computed addresses.
            strides: Tuple of stride values, one per dimension.
            max_values: Tuple of maximum values (exclusive) for each dimension.
        """
        self.strides = strides
        self.max_values = max_values
        self.base = base

    def get_addr_list(self) -> np.ndarray:
        """Generate addresses for all coordinate combinations.

        Computes the linear address for every possible combination of
        indices given the strides and max values.

        Returns:
            1D numpy array of addresses, one for each coordinate combination.
            Length equals product of all max_values.
        """
        indexs = list(np.ndindex(self.max_values))
        indexs = [np.array(index) for index in indexs]
        indexs = np.stack(indexs)  # [N, M]
        strides = np.array(self.strides)  # [M]
        addrs = einops.reduce(indexs * strides, "n m -> n", "sum") + self.base  # [N]
        return addrs


class TileMapping:
    """Defines tensor-to-memory tiling and rearrangement.

    Configures how a tensor should be rearranged for storage in a
    memory array. Uses einops formulas to specify the transformation
    between software (tensor) layout and hardware (memory) layout.

    Key concepts:
    - sw_einops: How to interpret the software (PyTorch) tensor dimensions
    - hw_einops: How to arrange dimensions in hardware (memory)
    - hw_temp_dim: Temporal dimensions (iterated sequentially)
    - hw_spat_dim: Spatial dimensions (contiguous in memory)

    Attributes:
        dtype (torch.dtype): Data type of tensor elements.
        sw_einops (str): Software einops pattern (e.g., "c h w -> c h w").
        hw_einops (str): Hardware einops pattern (e.g., "c h w -> (c h w)").
        hw_temp_dim (dict): Hardware temporal dimension sizes.
        hw_spat_dim (dict): Hardware spatial dimension sizes.
        base_addr (int): Base address in memory.
        strides (dict): Address strides per temporal dimension.

    Example:
        >>> import torch
        >>> from torchbit.tools import TileMapping
        >>>
        >>> # Simple 2D convolution input layout
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="c h w -> c h w",  # channel, height, width
        ...     hw_einops="c h w -> (c h w)",  # flatten to 1D in memory
        ...     hw_temp_dim={"c": 3},  # iterate over channels
        ...     hw_spat_dim={"h": 32, "w": 32},  # contiguous (h*w) elements
        ...     base_addr=0x1000,
        ...     strides={"c": 1024}  # each channel is 32*32*4 bytes = 4096 bits = 512 ints
        ... )
        >>>
        >>> # NHWC to linear memory for hardware
        >>> nhwc_mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="n c h w -> n c h w",  # batch, channel, height, width
        ...     hw_einops="n c h w -> (n c h w)",  # linearize
        ...     hw_temp_dim={"n": 1},  # process one batch at a time
        ...     hw_spat_dim={"c": 64, "h": 28, "w": 28},  # C*H*W elements
        ...     base_addr=0,
        ...     strides={"n": 64*28*28}  # stride for batch dimension
        ... )

    Common patterns:
        - Channel-first (NCHW): "c h w -> (c h w)"
        - Channel-last (NHWC): "c h w -> (h w c)"
        - Tiled: "c (h h_tile) (w w_tile) -> (h_tile w_tile c) h_tile w_tile"
    """

    dtype: torch.dtype
    sw_einops: str
    hw_einops: str
    hw_temp_dim: dict
    hw_spat_dim: dict
    base_addr: int = 0
    strides: dict = None

    def __post_init__(self):
        """Validate and compute derived attributes."""
        assert all(isinstance(k, str) for k in self.hw_temp_dim.keys()) and all(
            isinstance(v, int) for v in self.hw_temp_dim.values()
        ), "hw_temp_dim must be a dict of str to int"
        assert all(isinstance(k, str) for k in self.hw_spat_dim.keys()) and all(
            isinstance(v, int) for v in self.hw_spat_dim.values()
        ), "hw_spat_dim must be a dict of str to int"

        # Number of elements per temporal unit (spatial size)
        self.num = int(np.prod(list(self.hw_spat_dim.values())))
        self.sw_to_hw_formula = f"{self.sw_einops} -> {self.hw_einops}"
        self.hw_to_sw_formula = f"{self.hw_einops} -> {self.sw_einops}"

        if self.strides is not None:
            self.address_mapping = AddressMapping(
                self.base_addr,
                tuple(self.strides.values()),
                tuple(self.hw_temp_dim.values()),
            )


def tensor_to_cocotb_seq(tensor: torch.Tensor, mapping: TileMapping) -> list[int]:
    """Convert a tensor to a sequence of Cocotb-compatible integer values.

    Uses the TileMapping to rearrange the tensor and pack each row
    into an integer suitable for HDL interfaces.

    Args:
        tensor: Input PyTorch tensor.
        mapping: TileMapping defining the tensor-to-memory transformation.

    Returns:
        List of integers, one per temporal unit, ready for HDL assignment.

    Example:
        >>> tensor = torch.randn(3, 32, 32)  # 3 channels, 32x32
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="c h w -> c h w",
        ...     hw_einops="c h w -> (h w) c",  # NHWC-like
        ...     hw_temp_dim={"c": 3},
        ...     hw_spat_dim={"h": 32, "w": 32},
        ...     base_addr=0,
        ...     strides={"c": 1024}
        ... )
        >>> seq = tensor_to_cocotb_seq(tensor, mapping)
        >>> # seq[0] contains channel 0 data packed as integers
    """
    tensor_seq = einops.rearrange(
        tensor, mapping.sw_to_hw_formula, **mapping.hw_temp_dim, **mapping.hw_spat_dim
    )  # [N, M]
    return [Vector.from_tensor(tensor_row).to_cocotb() for tensor_row in tensor_seq]


def cocotb_seq_to_tensor(cocotb_seq: list[int], mapping: TileMapping) -> torch.Tensor:
    """Convert a sequence of Cocotb values back to a tensor.

    Unpacks each integer in the sequence and uses the TileMapping
    to rearrange into the original tensor layout.

    Args:
        cocotb_seq: List of integers from HDL interface.
        mapping: TileMapping defining the memory-to-tensor transformation.

    Returns:
        PyTorch tensor with the original layout.

    Example:
        >>> # Continuing from tensor_to_cocotb_seq example
        >>> result = cocotb_seq_to_tensor(seq, mapping)
        >>> # result.shape == (3, 32, 32)
    """
    tensor_seq = torch.stack(
        [
            Vector.from_cocotb(int_value, mapping.num, mapping.dtype).to_tensor()
            for int_value in cocotb_seq
        ]
    )
    tensor = einops.rearrange(
        tensor_seq,
        mapping.hw_to_sw_formula,
        **mapping.hw_temp_dim,
        **mapping.hw_spat_dim,
    )  # [N, M]
    return tensor
