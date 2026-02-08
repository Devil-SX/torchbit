"""
Address mapping utilities for hardware memory layouts.

Provides AddressMapping and ContiguousAddressMapping for computing linear
addresses from multi-dimensional coordinates using stride calculations.
"""
import numpy as np
import einops
from ..core.logic_sequence import LogicSequence


class AddressMapping:
    """Maps multi-dimensional indices to flat memory addresses.

    Computes linear addresses from multi-dimensional coordinates using
    stride calculations. This is useful for:
    - Row-major or column-major tensor storage
    - Strided memory layouts
    - Multi-bank memory partitioning

    The hw_temp_einops string defines the dimension ordering. Dimensions
    listed earlier correspond to higher-order address bits (outer loops),
    while dimensions listed later correspond to lower-order bits (inner loops).
    For example, "row col" means row varies slowest and col varies fastest.

    The hw_temp_dim and hw_temp_stride dicts can be passed in any key order;
    the canonical ordering is always determined by hw_temp_einops.

    Attributes:
        hw_temp_einops (str): Dimension ordering string (e.g., "row col").
        hw_temp_dim (dict): Dimension names to sizes.
        hw_temp_stride (dict): Dimension names to stride values.
        base (int): Base address offset.

    Example:
        >>> # Row-major 2D array (like C)
        >>> mapping = AddressMapping(
        ...     base=0x1000,
        ...     hw_temp_einops="row col",
        ...     hw_temp_dim={"row": 4, "col": 16},
        ...     hw_temp_stride={"row": 16, "col": 1},
        ... )
        >>> addrs = mapping.get_addr_list()
        >>>
        >>> # Column-major 2D array (like Fortran)
        >>> col_mapping = AddressMapping(
        ...     base=0,
        ...     hw_temp_einops="row col",
        ...     hw_temp_dim={"row": 4, "col": 4},
        ...     hw_temp_stride={"row": 1, "col": 4},
        ... )
    """

    def __init__(self, base: int, hw_temp_einops: str, hw_temp_dim: dict, hw_temp_stride: dict):
        """Initialize AddressMapping.

        Args:
            base: Base address offset added to all computed addresses.
            hw_temp_einops: Space-separated dimension names defining the
                iteration order. Earlier names = higher address bits.
                E.g., "row col" means row is outer loop, col is inner.
            hw_temp_dim: Dict mapping dimension names to sizes (exclusive).
                Keys must match names in hw_temp_einops.
            hw_temp_stride: Dict mapping dimension names to stride values.
                Keys must match names in hw_temp_einops.
        """
        # Parse dimension ordering from einops string
        dim_names = hw_temp_einops.split()

        assert set(dim_names) == set(hw_temp_dim.keys()), \
            f"hw_temp_einops dims {set(dim_names)} != hw_temp_dim keys {set(hw_temp_dim.keys())}"
        assert set(dim_names) == set(hw_temp_stride.keys()), \
            f"hw_temp_einops dims {set(dim_names)} != hw_temp_stride keys {set(hw_temp_stride.keys())}"

        self.hw_temp_einops = hw_temp_einops
        self._dim_names = dim_names
        # Reorder dicts to match hw_temp_einops ordering
        self.hw_temp_dim = {k: hw_temp_dim[k] for k in dim_names}
        self.hw_temp_stride = {k: hw_temp_stride[k] for k in dim_names}
        self.base = base

    def get_addr_list(self) -> LogicSequence:
        """Generate addresses for all coordinate combinations.

        Computes the linear address for every possible combination of
        indices given the strides and max values. Iteration follows the
        hw_temp_einops ordering (first dim = outermost loop).

        Returns:
            IntSequence of addresses, one for each coordinate combination.
            Length equals product of all dimension sizes.
        """
        max_values = tuple(self.hw_temp_dim[k] for k in self._dim_names)
        strides = tuple(self.hw_temp_stride[k] for k in self._dim_names)
        indexs = list(np.ndindex(max_values))
        indexs = [np.array(index) for index in indexs]
        indexs = np.stack(indexs)  # [N, M]
        strides_arr = np.array(strides)  # [M]
        addrs = einops.reduce(indexs * strides_arr, "n m -> n", "sum") + self.base  # [N]
        return LogicSequence(addrs.tolist())


class ContiguousAddressMapping(AddressMapping):
    """Address mapping with row-major contiguous strides auto-computed from dims.

    A convenience subclass of AddressMapping where strides are automatically
    inferred from the dimension sizes in row-major (C-style) order based on
    hw_temp_einops ordering. The last dimension in hw_temp_einops has stride 1,
    the second-to-last has stride equal to the last dimension's size, and so on.

    For hw_temp_einops="a b c" with dims {"a": 4, "b": 3, "c": 2},
    the auto-computed strides are: {"a": 6, "b": 2, "c": 1}.

    Attributes:
        hw_temp_einops (str): Dimension ordering string.
        hw_temp_dim (dict): Dimension names to sizes.
        hw_temp_stride (dict): Auto-computed row-major strides.
        base (int): Base address offset.

    Example:
        >>> mapping = ContiguousAddressMapping(
        ...     base=0x1000,
        ...     hw_temp_einops="row col",
        ...     hw_temp_dim={"row": 4, "col": 16},
        ... )
        >>> # Equivalent to AddressMapping(base=0x1000,
        >>> #     hw_temp_einops="row col",
        >>> #     hw_temp_dim={"row": 4, "col": 16},
        >>> #     hw_temp_stride={"row": 16, "col": 1})
        >>> addrs = mapping.get_addr_list()
    """

    def __init__(self, base: int, hw_temp_einops: str, hw_temp_dim: dict):
        """Initialize ContiguousAddressMapping with auto-computed strides.

        Args:
            base: Base address offset added to all computed addresses.
            hw_temp_einops: Space-separated dimension names defining the
                iteration order. Strides are computed in this order
                (last dim stride=1, row-major).
            hw_temp_dim: Dict mapping dimension names to sizes. Keys must
                match names in hw_temp_einops.
        """
        # Parse ordering from einops
        dim_names = hw_temp_einops.split()

        assert set(dim_names) == set(hw_temp_dim.keys()), \
            f"hw_temp_einops dims {set(dim_names)} != hw_temp_dim keys {set(hw_temp_dim.keys())}"

        # Compute row-major contiguous strides based on hw_temp_einops order
        strides = {}
        stride = 1
        for name in reversed(dim_names):
            strides[name] = stride
            stride *= hw_temp_dim[name]

        hw_temp_stride = {name: strides[name] for name in dim_names}

        super().__init__(base=base, hw_temp_einops=hw_temp_einops,
                         hw_temp_dim=hw_temp_dim, hw_temp_stride=hw_temp_stride)
