"""
1D tensor conversion utilities for hardware verification.

Provides the Vector class for converting between PyTorch 1D tensors and
HDL-compatible integer representations. Essential for driving and capturing
scalar and array signals in Cocotb testbenches.
"""
import cocotb.types
import torch
import numpy as np
import cocotb
from pathlib import Path
from .dtype import *
from ..utils.bit_ops import *


class Vector:
    """A 1D tensor wrapper for hardware verification.

    Vector is the fundamental data type for converting between PyTorch tensors
    and Cocotb-compatible formats. It handles 1D tensors and converts them
    to/from integer values suitable for driving HDL interfaces.

    The Vector class acts as the interface between PyTorch tensors and Cocotb
    LogicArrays or Verilog multi-bit interfaces. It supports:
    - Converting tensors to packed integers for driving HDL signals
    - Unpacking integers from HDL signals back to tensors
    - Direct conversion to/from Cocotb signal values

    Attributes:
        tensor (torch.Tensor): The underlying 1D PyTorch tensor.

    Raises:
        AssertionError: If tensor has more than 1 dimension.

    Example:
        >>> import torch
        >>> from torchbit.core import Vector
        >>>
        >>> # Convert tensor to cocotb value
        >>> tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        >>> vec = Vector.from_tensor(tensor)
        >>> cocotb_value = vec.to_cocotb()
        >>>
        >>> # Convert cocotb value back to tensor
        >>> vec = Vector.from_cocotb(cocotb_value, 3, torch.float32)
        >>> tensor = vec.to_tensor()

    Typical usage in Cocotb tests:
        >>> @cocotb.test
        >>> async def test_example(dut):
        >>>     # Drive inputs
        >>>     input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        >>>     dut.data_in.value = Vector.from_tensor(input_tensor).to_cocotb()
        >>>
        >>>     await RisingEdge(dut.clk)
        >>>
        >>>     # Capture outputs
        >>>     output_vec = Vector.from_cocotb(dut.data_out.value, 4, torch.float32)
        >>>     output_tensor = output_vec.to_tensor()
    """

    def __init__(self, tensor: torch.Tensor = None):
        """Initialize a Vector with a 1D tensor.

        Args:
            tensor: A 1D PyTorch tensor. Must have at most 1 dimension.

        Raises:
            AssertionError: If tensor has more than 1 dimension.
        """
        assert len(tensor.shape) <= 1
        self.tensor = tensor

    @staticmethod
    def from_int(value_int: int, num: int, dtype: torch.dtype) -> "Vector":
        """Create a Vector from an integer value by unpacking.

        Unpacks an integer into `num` elements of the specified dtype.
        Useful for reading HDL interface values back into tensor form.
        The least significant bits correspond to the first elements.

        Args:
            value_int: Integer value to unpack. Each element occupies
                bit_length bits where bit_length depends on dtype.
            num: Number of elements to create.
            dtype: PyTorch dtype (e.g., torch.float32, torch.int8).
                Must be a dtype supported by dtype_to_bits.

        Returns:
            A new Vector instance with the unpacked tensor.

        Raises:
            AssertionError: If value_int is not an int or dtype is unsupported.

        Example:
            >>> vec = Vector.from_int(0x41424142, 4, torch.uint8)
            >>> vec.to_tensor()
            tensor([66, 65, 66, 65], dtype=torch.uint8)
        """
        assert isinstance(value_int, int), "value must be an int, use Vector.from_int(value)"
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        vec = []
        mask = (1 << bit_length) - 1

        if num == 1:
            v = value_int & mask
            vec = np.array(v).astype(standard_numpy_dtype[bit_length])
            return Vector(torch.from_numpy(vec).view(dtype))
        else:
            for _ in range(num):
                vec.append(value_int & mask)
                value_int >>= bit_length
            vec = np.array(vec).astype(standard_numpy_dtype[bit_length])
            return Vector(torch.from_numpy(vec).view(dtype))

    @staticmethod
    def from_logic(value: cocotb.types.LogicArray | int, num: int, dtype: torch.dtype) -> "Vector":
        """Create a Vector from a logic value (LogicArray or int).

        Converts HDL signal values (LogicArray or integer) back into a Vector
        for comparison with PyTorch tensors. Handles X/Z states gracefully.

        Args:
            value: Cocotb LogicArray or int value from HDL interface.
                If a LogicArray contains X or Z states, zeros are returned.
            num: Number of elements expected in the resulting tensor.
            dtype: PyTorch dtype for the resulting tensor.

        Returns:
            A new Vector instance with the converted tensor.

        Raises:
            AssertionError: If value type is unsupported.

        Note:
            If the value contains X/Z states, a warning is printed and
            a tensor of zeros is returned.

        Example:
            >>> vec = Vector.from_logic(dut.data_out.value, 8, torch.float32)
            >>> tensor = vec.to_array()
        """
        assert isinstance(
            value, cocotb.types.LogicArray
        ) or isinstance(value, int), "value must be a cocotb logicarray value or int value, use Vector.from_logic(dut.io_xxx.value)"

        if isinstance(value, cocotb.types.LogicArray) and (("x" in value.binstr) or ("z" in value.binstr)):
            print("Warning: value is a X/Z value, use the zero result")
            return Vector(torch.zeros(num, dtype=dtype))

        value_int = int(value) if (isinstance(value, cocotb.types.LogicArray) or isinstance(value, cocotb.types.Logic)) else value
        return Vector.from_int(value_int, num, dtype)

    # Alias
    from_cocotb = from_logic

    @staticmethod
    def from_array(tensor: torch.Tensor) -> "Vector":
        """Create a Vector from a 1D PyTorch tensor (array).

        Alias for from_tensor(). This is the canonical name following
        the logic/array/matrix terminology.

        Args:
            tensor: A 1D PyTorch tensor of any supported dtype.

        Returns:
            A new Vector instance containing the tensor.

        Example:
            >>> tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
            >>> vec = Vector.from_array(tensor)
        """
        return Vector(tensor)

    # Alias
    from_tensor = from_array

    def to_logic(self) -> int:
        """Convert the Vector to a packed integer (logic value).

        Packs the tensor elements into a single integer suitable for
        driving HDL interfaces. For multi-element tensors, elements are
        packed with the last element at the most significant bits.

        Returns:
            An integer value for HDL signal assignment.

        Raises:
            AssertionError: If tensor has more than 1 dimension or dtype is unsupported.

        Example:
            >>> vec = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32))
            >>> dut.io_data.value = vec.to_logic()
        """
        assert len(self.tensor.shape) <= 1
        assert self.tensor.dtype in dtype_to_bits.keys()
        tensor = self.tensor

        bit_length = dtype_to_bits[tensor.dtype]
        tensor = tensor.view(standard_torch_dtype[bit_length])
        tensor_numpy = tensor.numpy()

        result = 0
        mask = (1 << bit_length) - 1
        if len(self.tensor.shape) == 1:
            for v in reversed(tensor_numpy.astype(object)):
                result = (result << bit_length) | (v & mask)
        else:
            result = tensor_numpy.item() & mask
        return result

    def to_cocotb(self) -> int:
        """Alias for to_logic(). Pack the tensor into a single integer."""
        return self.to_logic()

    def to_int(self) -> int:
        """Alias for to_logic(). Pack the tensor into a single integer."""
        return self.to_logic()

    def to_array(self) -> torch.Tensor:
        """Get the underlying PyTorch tensor (array).

        Returns:
            The 1D PyTorch tensor stored in this Vector.

        Example:
            >>> vec = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32))
            >>> tensor = vec.to_array()
        """
        return self.tensor

    def to_tensor(self) -> torch.Tensor:
        """Alias for to_array(). Get the underlying PyTorch tensor.

        Returns:
            The 1D PyTorch tensor stored in this Vector.
        """
        return self.tensor


def array_to_logic(tensor: torch.Tensor) -> int:
    """Convert a 1D tensor (array) directly to a packed integer (logic value).

    This is a convenience wrapper combining Vector.from_array() and to_logic().

    Args:
        tensor: A 1D PyTorch tensor of any supported dtype.

    Returns:
        An integer value for HDL signal assignment.

    Example:
        >>> import torch
        >>> from torchbit.core import array_to_logic
        >>> tensor = torch.tensor([1.5, 2.5], dtype=torch.float32)
        >>> dut.io_din.value = array_to_logic(tensor)
    """
    return Vector.from_array(tensor).to_logic()


def logic_to_array(value: cocotb.types.LogicArray | int, num: int, dtype: torch.dtype) -> torch.Tensor:
    """Convert a logic value (LogicArray or int) directly to a 1D tensor (array).

    This is a convenience wrapper combining Vector.from_logic() and to_array().

    Args:
        value: Cocotb LogicArray or int value from HDL interface.
        num: Number of elements expected in the resulting tensor.
        dtype: PyTorch dtype for the resulting tensor.

    Returns:
        A 1D PyTorch tensor with the converted values.

    Example:
        >>> from torchbit.core import logic_to_array
        >>> tensor = logic_to_array(dut.io_dout.value, 8, torch.float32)
    """
    return Vector.from_logic(value, num, dtype).to_array()


# Aliases
tensor_to_cocotb = array_to_logic
cocotb_to_tensor = logic_to_array
