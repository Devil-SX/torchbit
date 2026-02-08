"""
2D tensor conversion utilities for memory-mapped hardware verification.

Provides the VectorSequence class for converting between PyTorch 2D tensors and
memory files (hex/bin) used with Verilog $readmemh/$writememh and similar
system tasks. Essential for verifying memory-mapped IP blocks and loading
initial memory contents.
"""
import cocotb.types
import torch
import numpy as np
import cocotb
from pathlib import Path
from .dtype import *
from ..utils.bit_ops import *


def read_arr(bytes_data: bytes, bit_length: int, endianess: str = "little") -> np.ndarray:
    """Convert bytes to a numpy array with specified bit length.

    Reads raw byte data and converts it to a numpy array of the specified
    bit width. Handles endianness conversion between the byte data and
    the resulting array representation.

    Args:
        bytes_data: Raw byte data to convert.
        bit_length: Bit width of each element (8, 16, 32, 64).
        endianess: Byte order of the input data. Either "little" or "big".
            Defaults to "little".

    Returns:
        A numpy array with the converted values.

    Raises:
        AssertionError: If endianess is not "little" or "big".

    Example:
        >>> import torch
        >>> data = bytes.fromhex("41424344")
        >>> arr = read_arr(data, 8, "little")
        >>> # arr contains [0x44, 0x43, 0x42, 0x41] for little-endian
    """
    assert endianess in ["little", "big"], f"endianess must be 'little' or 'big', got {endianess}"
    # bigendian bytes -> numpy little end array
    numpy_dtype = standard_numpy_dtype[bit_length]
    if endianess == "big":
        be_numpy_dtype = standard_be_numpy_dtype[bit_length]
    else:
        be_numpy_dtype = standard_le_numpy_dtype[bit_length]
    arr = np.frombuffer(bytes_data, dtype=be_numpy_dtype)
    arr = arr.astype(numpy_dtype)
    if endianess == "big":
        arr = np.flip(arr, 0).copy()
    else:
        arr = arr.copy()
    return arr


def to_bytes(arr: np.ndarray, endianess: str = "little") -> bytes:
    """Convert a numpy array to bytes with specified bit length.

    Converts a numpy array to raw byte data with proper endianness handling.
    The array is first converted to the appropriate big-endian format for
    output.

    Args:
        arr: 1D numpy array to convert.
        endianess: Byte order for the output. Either "little" or "big".
            Defaults to "little".

    Returns:
        Raw byte data representing the array.

    Raises:
        AssertionError: If endianess is not "little" or "big", or if array
            has more than 1 dimension, or if dtype is unsupported.
    """
    assert endianess in ["little", "big"], f"endianess must be 'little' or 'big', got {endianess}"
    # numpy little end array -> bigendian bytes
    assert len(arr.shape) <= 1
    dtype = arr.dtype
    assert dtype in numpy_dtype_to_bits.keys(), f"{dtype} not in {numpy_dtype_to_bits.keys()}"

    bit_length = numpy_dtype_to_bits[dtype]
    if endianess == "big":
        be_numpy_dtype = standard_be_numpy_dtype[bit_length]
    else:
        be_numpy_dtype = standard_le_numpy_dtype[bit_length]
    arr = arr.copy()

    if endianess == "big":
        arr = np.flip(arr, 0).astype(be_numpy_dtype)
    else:
        arr = arr.astype(be_numpy_dtype)
    return arr.tobytes()


class VectorSequence:
    """A 2D tensor wrapper for memory-mapped hardware verification.

    VectorSequence handles 2D tensors for verifying memory-mapped interfaces.
    It supports reading/writing to memory hex files and binary files
    commonly used with Verilog $readmemh/$writememh.

    The VectorSequence class is designed for:
    - Loading initial memory contents from files
    - Dumping simulation results to files
    - Verifying memory-mapped IP blocks
    - Creating test vectors for memory-based accelerators

    Attributes:
        tensor (torch.Tensor): The underlying 2D PyTorch tensor.

    Raises:
        AssertionError: If tensor does not have exactly 2 dimensions.

    Example:
        >>> import torch
        >>> from torchbit.core import VectorSequence
        >>>
        >>> # Load from memory hex file
        >>> vs = VectorSequence.from_memhexfile("memory.hex", torch.float32)
        >>> print(vs.tensor.shape)  # e.g., torch.Size([256, 4])
        >>>
        >>> # Dump to memory hex file
        >>> vs.to_memhexfile("output.hex")
        >>>
        >>> # Create from tensor
        >>> tensor = torch.randn(256, 4)
        >>> vs = VectorSequence.from_tensor(tensor)

    Typical usage in Cocotb tests:
        >>> @cocotb.test
        >>> async def test_memory(dut):
        >>>     # Load memory contents
        >>>     mem_vs = VectorSequence.from_memhexfile("init.hex", torch.float32)
        >>>     await dut.mem_init.value  # Signal that memory is ready
        >>>
        >>>     # Read memory through interface
        >>>     results = []
        >>>     for addr in range(256):
        >>>         dut.addr.value = addr
        >>>         await RisingEdge(dut.clk)
        >>>         results.append(dut.data_out.value)
    """

    def __init__(self, tensor: torch.Tensor = None):
        """Initialize a VectorSequence with a 2D tensor.

        Args:
            tensor: A 2D PyTorch tensor. Must have exactly 2 dimensions.

        Raises:
            AssertionError: If tensor does not have exactly 2 dimensions.
        """
        assert len(tensor.shape) == 2
        self.tensor = tensor

    @staticmethod
    def from_memhexfile(in_path: str | Path, dtype: torch.dtype, endianess: str = "little") -> "VectorSequence":
        """Create a VectorSequence by reading a memory hex file.

        Reads a file in $readmemh format and converts to a 2D tensor.
        Each line in the hex file becomes one row in the VectorSequence.

        The hex file format expects one hexadecimal value per line,
        optionally with underscores for readability.

        Args:
            in_path: Path to the hex file. Each line should contain
                a hexadecimal value (e.g., "3f800000").
            dtype: PyTorch dtype for tensor elements.
            endianess: Byte order of the hex data. Either "little" or "big".
                Defaults to "little".

        Returns:
            A new VectorSequence instance with the loaded tensor.

        Raises:
            AssertionError: If dtype is unsupported or file format is invalid.

        Example:
            >>> vs = VectorSequence.from_memhexfile("data.hex", torch.float32)
            >>> # hex file contents:
            >>> # 00000000
            >>> # 3f800000
            >>> # 40000000
            >>> # ...
        """
        # get the bit length of dtype
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        # read a memhex file and convert to tensor
        with open(in_path, "r") as f:
            lines = f.readlines()

        tensor_list = []
        for line in lines:
            byte_data = bytes.fromhex(line.strip())
            arr = read_arr(byte_data, bit_length, endianess)
            tensor_row = torch.from_numpy(arr)
            tensor_list.append(tensor_row)

        tensor = torch.stack(tensor_list)
        return VectorSequence(tensor.view(dtype))

    @staticmethod
    def from_binfile(in_path: str | Path, num: int, dtype: torch.dtype, endianess: str = "little") -> "VectorSequence":
        """Create a VectorSequence by reading a binary file.

        Reads raw binary data and converts to a 2D tensor.
        Each read operation produces `num` elements per row.

        Args:
            in_path: Path to the binary file.
            num: Number of elements per row. The file size should be
                divisible by (num * bit_length / 8).
            dtype: PyTorch dtype for tensor elements.
            endianess: Byte order of the binary data. Either "little" or "big".
                Defaults to "little".

        Returns:
            A new VectorSequence instance with the loaded tensor.

        Raises:
            AssertionError: If dtype is unsupported.

        Example:
            >>> # Read binary data, 16 elements per row
            >>> vs = VectorSequence.from_binfile("data.bin", num=16, dtype=torch.float32)
        """
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        # read a binary file and convert to tensor
        with open(in_path, "rb") as f:
            tensor_list = []
            while True:
                byte_data = f.read(bit_length * num // 8)
                if not byte_data:
                    break
                arr = read_arr(byte_data, bit_length, endianess)
                tensor_row = torch.from_numpy(arr)
                tensor_list.append(tensor_row)

        tensor = torch.stack(tensor_list)
        return VectorSequence(tensor.view(dtype))

    def to_memhexfile(self, out_path: str | Path, endianess: str = "little") -> None:
        """Write the VectorSequence to a memory hex file.

        Dumps the 2D tensor to a file in $readmemh compatible format.
        Each tensor row becomes one line in the hex file containing
        the hexadecimal representation of that row.

        Args:
            out_path: Path to write the hex file. Parent directories
                will be created if they don't exist.
            endianess: Byte order for the output. Either "little" or "big".
                Defaults to "little".

        Raises:
            AssertionError: If tensor is not 2D or dtype is unsupported.

        Example:
            >>> vs.to_memhexfile("output.hex")
            >>> # produces:
            >>> # 00000000
            >>> # 3f800000
            >>> # 40000000
            >>> # ...
        """
        # load a tensor and save as memhex that verilog could read
        # assert is a 2D tensor
        tensor = self.tensor

        assert len(tensor.shape) == 2
        assert tensor.dtype in dtype_to_bits.keys()
        # get the bit length of dtype

        bit_length = dtype_to_bits[tensor.dtype]
        tensor = tensor.view(standard_torch_dtype[bit_length])
        tensor_numpy = tensor.numpy()

        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for tensor_row in tensor_numpy:
                hex_row = to_bytes(tensor_row, endianess).hex()
                f.write(hex_row + "\n")

    def to_binfile(self, out_path: str | Path, endianess: str = "little") -> None:
        """Write the VectorSequence to a binary file.

        Dumps the 2D tensor to a raw binary file.
        Each tensor row is written as a sequence of bytes.

        Args:
            out_path: Path to write the binary file. Parent directories
                will be created if they don't exist.
            endianess: Byte order for the output. Either "little" or "big".
                Defaults to "little".

        Raises:
            AssertionError: If tensor is not 2D or dtype is unsupported.

        Example:
            >>> vs.to_binfile("output.bin")
        """
        # load a tensor and save as binary that verilog could read
        # assert is a 2D tensor
        tensor = self.tensor

        assert len(tensor.shape) == 2
        assert tensor.dtype in dtype_to_bits.keys()
        # get the bit length of dtype

        bit_length = dtype_to_bits[tensor.dtype]
        tensor = tensor.view(standard_torch_dtype[bit_length])
        tensor_numpy = tensor.numpy()

        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for tensor_row in tensor_numpy:
                byte_row = to_bytes(tensor_row, endianess)
                f.write(byte_row)

    def to_logic_sequence(self):
        """Convert each row to a packed integer via Vector, returning a LogicSequence.

        Returns:
            LogicSequence where each element is the packed integer of one row.
        """
        from .logic_sequence import LogicSequence
        from .vector import Vector
        return LogicSequence(Vector.from_array(row).to_logic() for row in self.tensor)

    def to_int_sequence(self):
        """Alias for to_logic_sequence()."""
        return self.to_logic_sequence()

    @staticmethod
    def from_logic_sequence(logic_seq, num: int, dtype: torch.dtype) -> "VectorSequence":
        """Create a VectorSequence from a LogicSequence via Vector.

        Args:
            logic_seq: LogicSequence (or list of ints) to convert.
            num: Number of elements per row.
            dtype: PyTorch dtype for tensor elements.

        Returns:
            A new VectorSequence instance.
        """
        from .vector import Vector
        rows = [Vector.from_logic(v, num, dtype).to_array() for v in logic_seq]
        return VectorSequence(torch.stack(rows))

    # Alias
    from_int_sequence = from_logic_sequence

    def to_matrix(self) -> torch.Tensor:
        """Get the underlying PyTorch tensor (matrix).

        Returns:
            The 2D PyTorch tensor stored in this VectorSequence.

        Example:
            >>> vs = VectorSequence.from_tensor(torch.randn(256, 4))
            >>> tensor = vs.to_matrix()
        """
        return self.tensor

    def to_tensor(self) -> torch.Tensor:
        """Alias for to_matrix(). Get the underlying PyTorch tensor.

        Returns:
            The 2D PyTorch tensor stored in this VectorSequence.
        """
        return self.tensor

    @staticmethod
    def from_matrix(tensor: torch.Tensor) -> "VectorSequence":
        """Create a VectorSequence from a 2D PyTorch tensor (matrix).

        Alias for from_tensor(). This is the canonical name following
        the logic/array/matrix terminology.

        Args:
            tensor: A 2D PyTorch tensor of any supported dtype.

        Returns:
            A new VectorSequence instance containing the tensor.

        Example:
            >>> tensor = torch.randn(256, 4)
            >>> vs = VectorSequence.from_matrix(tensor)
        """
        return VectorSequence(tensor)

    # Alias
    from_tensor = from_matrix


# Backward compatibility alias
Matrix = VectorSequence
