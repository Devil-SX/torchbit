"""
Tensor-to-memory mapping and address translation utilities.

Provides TileMapping and AddressMapping classes for configuring how tensors
should be arranged in memory. Essential for deep learning accelerator
verification where tensor data must be tiled, strided, and mapped to
specific memory layouts.
"""
import torch
import numpy as np
import einops
from ..core.vector import Vector
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
        >>>
        >>> # Column-major 2D array (like Fortran)
        >>> col_mapping = AddressMapping(
        ...     base=0,
        ...     strides=(1, 4),
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


def _is_2d_hw_format(pattern: str) -> bool:
    """Check if pattern is a valid 2D hardware matrix format.

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

    Configures how a tensor should be rearranged for storage in a
    memory array. Uses einops-style dimension notation to specify
    the transformation between software (tensor) layout and hardware
    (2D matrix) layout.

    Key concepts:
    - sw_einops: Software tensor dimension layout (e.g., "(ht hs) (wt ws) ct")
    - hw_einops: Hardware 2D matrix layout (e.g., "(ht wt ct) (hs ws)")
    - hw_temp_dim: Temporal dimension sizes (first group of hw_einops)
    - hw_spat_dim: Spatial dimension sizes (second group of hw_einops)

    Attributes:
        dtype (torch.dtype): Data type of tensor elements.
        sw_einops (str): Software dimension layout without "->".
        hw_einops (str): Hardware 2D matrix layout "(temporal) (spatial)".
        hw_temp_dim (dict): Hardware temporal dimension sizes.
        hw_spat_dim (dict): Hardware spatial dimension sizes.
        base_addr (int): Base address in memory.
        strides (dict): Address strides per temporal dimension.

    Example:
        >>> # Example: Transform (ht hs) (wt ws) ct -> (ht wt ct) (hs ws)
        >>> # Software: 4D tensor with grouped dimensions
        >>> # Hardware: 2D matrix where first group is temporal, second is spatial
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="(ht hs) (wt ws) ct",
        ...     hw_einops="(ht wt ct) (hs ws)",
        ...     hw_temp_dim={"ht": 2, "wt": 2, "ct": 3},
        ...     hw_spat_dim={"hs": 4, "ws": 4},
        ...     base_addr=0x1000,
        ...     strides={"ht": 6, "wt": 3, "ct": 1}
        ... )

    Simple example:
        >>> # 3D tensor (c, h, w) -> 2D matrix (c, (h w))
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="c h w",
        ...     hw_einops="c (h w)",  # c=temporal, (h w)=spatial
        ...     hw_temp_dim={"c": 3},
        ...     hw_spat_dim={"h": 8, "w": 8},
        ...     base_addr=0,
        ...     strides={"c": 64}
        ... )
        >>> tensor = torch.randn(3, 8, 8)
        >>> values, addrs = mapping.to_hw(tensor)
        >>> tensor_restored = mapping.to_sw(values, addrs)
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

        if self.strides is not None:
            self.address_mapping = AddressMapping(
                self.base_addr,
                tuple(self.strides.values()),
                tuple(self.hw_temp_dim.values()),
            )

    def to_hw(self, tensor: torch.Tensor) -> tuple[list[int], list[int]]:
        """Convert a software tensor to hardware format.

        Transforms the tensor according to the TileMapping and produces
        two lists: values (packed integers) and their corresponding addresses.

        Args:
            tensor: Input PyTorch tensor with shape matching sw_einops pattern.

        Returns:
            A tuple of (values_list, addresses_list):
            - values_list: List of packed integers, one per temporal unit.
            - addresses_list: List of memory addresses for each value.

        Raises:
            AssertionError: If strides is not set (address_mapping required).

        Example:
            >>> mapping = TileMapping(
            ...     dtype=torch.float32,
            ...     sw_einops="c h w",
            ...     hw_einops="c (h w)",
            ...     hw_temp_dim={"c": 3},
            ...     hw_spat_dim={"h": 32, "w": 32},
            ...     base_addr=0x1000,
            ...     strides={"c": 1024}
            ... )
            >>> tensor = torch.randn(3, 32, 32)
            >>> values, addrs = mapping.to_hw(tensor)
            >>> # len(values) == 3, len(addrs) == 3
        """
        assert hasattr(self, 'address_mapping'), \
            "to_hw() requires strides to be set for address generation"

        # Get address list from address_mapping
        addr_list = self.address_mapping.get_addr_list()

        # Rearrange tensor from software layout to hardware 2D matrix layout
        tensor_seq = einops.rearrange(
            tensor,
            self.sw_to_hw_formula,
            **self.hw_temp_dim,
            **self.hw_spat_dim,
        )  # [N, M] where N is temporal iterations, M is spatial elements

        # Convert each temporal row to packed integer
        values_list = [Vector.from_tensor(tensor_row).to_cocotb() for tensor_row in tensor_seq]

        # Convert numpy array to list of ints
        addresses_list = addr_list.tolist()

        return values_list, addresses_list

    def to_sw(self, values_list: list[int], addresses_list: list[int]) -> torch.Tensor:
        """Convert hardware format back to software tensor.

        Takes values and addresses from hardware and reconstructs the
        original tensor layout according to the TileMapping.

        Args:
            values_list: List of packed integers from hardware.
            addresses_list: List of corresponding memory addresses.

        Returns:
            A PyTorch tensor with shape matching sw_einops pattern.

        Example:
            >>> mapping = TileMapping(
            ...     dtype=torch.float32,
            ...     sw_einops="c h w",
            ...     hw_einops="c (h w)",
            ...     hw_temp_dim={"c": 3},
            ...     hw_spat_dim={"h": 32, "w": 32},
            ...     base_addr=0x1000,
            ...     strides={"c": 1024}
            ... )
            >>> values, addrs = mapping.to_hw(tensor)
            >>> tensor_restored = mapping.to_sw(values, addrs)
            >>> assert torch.allclose(tensor, tensor_restored)
        """
        # Convert each integer back to tensor row
        tensor_seq = [
            Vector.from_cocotb(int_value, self.num, self.dtype).to_tensor()
            for int_value in values_list
        ]

        # Stack and rearrange from hardware layout to software layout
        tensor = einops.rearrange(
            tensor_seq,
            self.hw_to_sw_formula,
            **self.hw_temp_dim,
            **self.hw_spat_dim,
        )

        return tensor


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
        ...     sw_einops="c h w",
        ...     hw_einops="c (h w)",
        ...     hw_temp_dim={"c": 3},
        ...     hw_spat_dim={"h": 32, "w": 32},
        ...     base_addr=0,
        ...     strides=None
        ... )
        >>> seq = tensor_to_cocotb_seq(tensor, mapping)
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
        >>> result = cocotb_seq_to_tensor(seq, mapping)
        >>> # result.shape matches original tensor shape
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
    )
    return tensor
