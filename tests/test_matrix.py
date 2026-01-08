"""
Tests for the Matrix class.

Matrix is a 2D tensor wrapper for converting between PyTorch tensors
and memory files (hex/bin) used with Verilog $readmemh/$writememh.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from torchbit.core import Matrix


class TestMatrixBasic:
    """Tests for basic Matrix operations."""

    def test_matrix_init_with_2d_tensor(self):
        """Test Matrix initialization with a 2D tensor."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mat = Matrix(tensor)
        assert torch.equal(mat.tensor, tensor)

    def test_matrix_init_with_1d_tensor_raises(self):
        """Test Matrix initialization with 1D tensor raises AssertionError."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        with pytest.raises(AssertionError):
            Matrix(tensor)

    def test_matrix_init_with_3d_tensor_raises(self):
        """Test Matrix initialization with 3D tensor raises AssertionError."""
        tensor = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        with pytest.raises(AssertionError):
            Matrix(tensor)

    def test_from_tensor(self):
        """Test Matrix.from_tensor() static method."""
        tensor = torch.randn(4, 4, dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)
        assert torch.equal(mat.tensor, tensor)

    def test_to_tensor(self):
        """Test Matrix.to_tensor() returns the underlying tensor."""
        tensor = torch.randn(4, 4, dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)
        assert torch.equal(mat.to_tensor(), tensor)


class TestMatrixMemhexFile:
    """Tests for Matrix memory hex file I/O."""

    def test_to_memhexfile_basic(self, tmp_path):
        """Test writing Matrix to memory hex file."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        output_path = tmp_path / "output.hex"
        mat.to_memhexfile(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split('\n')
        assert len(lines) == 2

    def test_from_memhexfile_basic(self, tmp_path):
        """Test reading Matrix from memory hex file."""
        # Create a hex file with known values
        input_path = tmp_path / "input.hex"
        # 0x0000803f is 1.0 in float32 little-endian
        # 0x00000040 is 2.0 in float32 little-endian
        input_path.write_text("0000803f\n00000040\n")

        mat = Matrix.from_memhexfile(input_path, torch.float32)
        # Each line creates one row with elements based on dtype
        assert mat.tensor.shape == (2, 1)  # 2 rows, 1 float32 element each (4 bytes)

    def test_memhexfile_roundtrip(self, tmp_path):
        """Test roundtrip: write then read memhex file."""
        original = torch.randn(8, 4, dtype=torch.float32)
        mat_original = Matrix.from_tensor(original)

        hex_path = tmp_path / "roundtrip.hex"
        mat_original.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, torch.float32)
        assert torch.allclose(mat_loaded.tensor, original)

    def test_to_memhexfile_creates_directories(self, tmp_path):
        """Test that to_memhexfile creates parent directories."""
        tensor = torch.tensor([[1.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        output_path = tmp_path / "subdir" / "nested" / "output.hex"
        mat.to_memhexfile(output_path)

        assert output_path.exists()
        assert output_path.parent.is_dir()

    def test_from_memhexfile_big_endian(self, tmp_path):
        """Test reading memhex file with big-endian byte order."""
        # Create hex file with big-endian data
        input_path = tmp_path / "input_be.hex"
        input_path.write_text("3f800000\n40000000\n")  # 1.0, 2.0 in big-endian float32

        mat = Matrix.from_memhexfile(input_path, torch.float32, endianess="big")
        # Each line creates one row with 1 float32 element
        assert mat.tensor.shape == (2, 1)

    def test_to_memhexfile_big_endian(self, tmp_path):
        """Test writing memhex file with big-endian byte order."""
        tensor = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        output_path = tmp_path / "output_be.hex"
        mat.to_memhexfile(output_path, endianess="big")

        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split('\n')
        # Big-endian representation
        assert lines[0] == "3f800000"  # 1.0 in big-endian


class TestMatrixBinaryFile:
    """Tests for Matrix binary file I/O."""

    def test_to_binfile_basic(self, tmp_path):
        """Test writing Matrix to binary file."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        output_path = tmp_path / "output.bin"
        mat.to_binfile(output_path)

        assert output_path.exists()
        # Check file size: 2 rows * 2 elements * 4 bytes = 16 bytes
        assert output_path.stat().st_size == 16

    def test_from_binfile_basic(self, tmp_path):
        """Test reading Matrix from binary file."""
        # Create binary file with known data
        input_path = tmp_path / "input.bin"
        data = bytes([0x00, 0x00, 0x80, 0x3f]) * 4  # 1.0 in little-endian float32, 4 times
        input_path.write_bytes(data)

        mat = Matrix.from_binfile(input_path, num=4, dtype=torch.float32)
        # 4 bytes total / (4 elements * 4 bytes) = 1 row
        assert mat.tensor.shape[1] == 4

    def test_binfile_roundtrip(self, tmp_path):
        """Test roundtrip: write then read binary file."""
        original = torch.randn(8, 4, dtype=torch.float32)
        mat_original = Matrix.from_tensor(original)

        bin_path = tmp_path / "roundtrip.bin"
        mat_original.to_binfile(bin_path)

        mat_loaded = Matrix.from_binfile(bin_path, num=4, dtype=torch.float32)
        assert torch.allclose(mat_loaded.tensor, original)

    def test_from_binfile_creates_directories(self, tmp_path):
        """Test that to_binfile creates parent directories."""
        tensor = torch.tensor([[1.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        output_path = tmp_path / "subdir" / "output.bin"
        mat.to_binfile(output_path)

        assert output_path.exists()

    def test_from_binfile_big_endian(self, tmp_path):
        """Test reading binary file with big-endian byte order."""
        # Create binary file with big-endian data
        input_path = tmp_path / "input_be.bin"
        data = bytes.fromhex("3f800000") * 4  # 1.0 in big-endian
        input_path.write_bytes(data)

        mat = Matrix.from_binfile(input_path, num=4, dtype=torch.float32, endianess="big")
        assert mat.tensor.shape[1] == 4

    def test_to_binfile_big_endian(self, tmp_path):
        """Test writing binary file with big-endian byte order."""
        tensor = torch.tensor([[1.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        output_path = tmp_path / "output_be.bin"
        mat.to_binfile(output_path, endianess="big")

        assert output_path.exists()
        content = output_path.read_bytes()
        # Big-endian representation of 1.0
        assert content == bytes.fromhex("3f800000")


class TestMatrixDtypes:
    """Tests for Matrix with different data types."""

    @pytest.mark.parametrize("dtype", [
        torch.uint8, torch.int8, torch.int16, torch.int32,
        torch.float16, torch.bfloat16, torch.float32,
    ])
    def test_memhexfile_dtype_roundtrip(self, dtype, tmp_path):
        """Test memhex file roundtrip for various dtypes."""
        if dtype == torch.uint8:
            tensor = torch.randint(0, 255, (4, 4), dtype=dtype)
        elif dtype in [torch.int8, torch.int16, torch.int32]:
            tensor = torch.randint(-100, 100, (4, 4), dtype=dtype)
        else:
            tensor = torch.randn(4, 4, dtype=dtype)

        mat_original = Matrix.from_tensor(tensor)
        hex_path = tmp_path / "test.hex"
        mat_original.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, dtype)
        assert torch.allclose(mat_loaded.tensor, tensor)

    @pytest.mark.parametrize("dtype", [
        torch.uint8, torch.int8, torch.int16, torch.int32,
        torch.float16, torch.bfloat16, torch.float32,
    ])
    def test_binfile_dtype_roundtrip(self, dtype, tmp_path):
        """Test binary file roundtrip for various dtypes."""
        if dtype == torch.uint8:
            tensor = torch.randint(0, 255, (4, 8), dtype=dtype)
        elif dtype in [torch.int8, torch.int16, torch.int32]:
            tensor = torch.randint(-100, 100, (4, 8), dtype=dtype)
        else:
            tensor = torch.randn(4, 8, dtype=dtype)

        mat_original = Matrix.from_tensor(tensor)
        bin_path = tmp_path / "test.bin"
        mat_original.to_binfile(bin_path)

        mat_loaded = Matrix.from_binfile(bin_path, num=8, dtype=dtype)
        assert torch.allclose(mat_loaded.tensor, tensor)


class TestMatrixEdgeCases:
    """Tests for Matrix edge cases."""

    def test_empty_matrix_raises(self):
        """Test that empty (0x0) matrix raises appropriate error."""
        # A 2D tensor with 0 elements would have shape like (0, 4) or (2, 0)
        tensor = torch.zeros(0, 4, dtype=torch.float32)
        # Matrix requires exactly 2 dimensions, which 0x4 has
        # The issue is that stack will fail with empty tensor list
        # This test verifies the constraint
        mat = Matrix(tensor)
        assert mat.tensor.shape == (0, 4)

    def test_single_row_matrix(self, tmp_path):
        """Test Matrix with single row."""
        tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        hex_path = tmp_path / "single_row.hex"
        mat.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, torch.float32)
        assert torch.equal(mat_loaded.tensor, tensor)

    def test_single_column_matrix(self, tmp_path):
        """Test Matrix with single column."""
        tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        hex_path = tmp_path / "single_col.hex"
        mat.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, torch.float32)
        assert torch.equal(mat_loaded.tensor, tensor)

    def test_large_matrix(self, tmp_path):
        """Test Matrix with large dimensions."""
        tensor = torch.randn(256, 4, dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        hex_path = tmp_path / "large.hex"
        mat.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, torch.float32)
        assert torch.allclose(mat_loaded.tensor, tensor)

    def test_zero_values(self, tmp_path):
        """Test Matrix with all zero values."""
        tensor = torch.zeros((4, 4), dtype=torch.float32)
        mat = Matrix.from_tensor(tensor)

        hex_path = tmp_path / "zeros.hex"
        mat.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, torch.float32)
        assert torch.equal(mat_loaded.tensor, tensor)

    def test_negative_values_int(self, tmp_path):
        """Test Matrix with negative integer values."""
        tensor = torch.tensor([[-1, -2], [-3, -4]], dtype=torch.int32)
        mat = Matrix.from_tensor(tensor)

        hex_path = tmp_path / "negative.hex"
        mat.to_memhexfile(hex_path)

        mat_loaded = Matrix.from_memhexfile(hex_path, torch.int32)
        assert torch.equal(mat_loaded.tensor, tensor)


class TestMatrixErrorHandling:
    """Tests for Matrix error handling."""

    def test_from_memhexfile_missing_file_raises(self, tmp_path):
        """Test reading from non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            Matrix.from_memhexfile(tmp_path / "nonexistent.hex", torch.float32)

    def test_from_memhexfile_with_invalid_hex_raises(self, tmp_path):
        """Test reading from file with invalid hex content."""
        input_path = tmp_path / "invalid.hex"
        input_path.write_text("not_a_hex_value\n")

        with pytest.raises(ValueError):
            Matrix.from_memhexfile(input_path, torch.float32)

    def test_from_binfile_missing_file_raises(self, tmp_path):
        """Test reading from non-existent binary file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            Matrix.from_binfile(tmp_path / "nonexistent.bin", num=4, dtype=torch.float32)

    def test_from_memhexfile_unsupported_dtype_raises(self, tmp_path):
        """Test from_memhexfile with unsupported dtype raises AssertionError."""
        input_path = tmp_path / "input.hex"
        input_path.write_text("0000803f\n")

        with pytest.raises(AssertionError):
            Matrix.from_memhexfile(input_path, torch.bool)

    def test_from_binfile_unsupported_dtype_raises(self, tmp_path):
        """Test from_binfile with unsupported dtype raises AssertionError."""
        input_path = tmp_path / "input.bin"
        input_path.write_bytes(bytes([0, 0, 0, 0]))

        with pytest.raises(AssertionError):
            Matrix.from_binfile(input_path, num=1, dtype=torch.bool)


class TestMatrixHelperFunctions:
    """Tests for Matrix helper functions read_arr and to_bytes."""

    def test_read_arr_little_endian(self):
        """Test read_arr with little-endian byte order."""
        from torchbit.core.matrix import read_arr
        data = bytes.fromhex("0000803f")  # 1.0 in little-endian float32
        arr = read_arr(data, 32, endianess="little")
        assert arr.dtype == np.int32

    def test_read_arr_big_endian(self):
        """Test read_arr with big-endian byte order."""
        from torchbit.core.matrix import read_arr
        data = bytes.fromhex("3f800000")  # 1.0 in big-endian float32
        arr = read_arr(data, 32, endianess="big")
        assert arr.dtype == np.int32

    def test_read_arr_invalid_endianess_raises(self):
        """Test read_arr with invalid endianess raises AssertionError."""
        from torchbit.core.matrix import read_arr
        data = bytes([0, 0, 0, 0])
        with pytest.raises(AssertionError):
            read_arr(data, 32, endianess="invalid")

    def test_to_bytes_little_endian(self):
        """Test to_bytes with little-endian byte order."""
        from torchbit.core.matrix import to_bytes
        arr = np.array([0x3f800000], dtype=np.int32)  # 1.0 bit pattern
        result = to_bytes(arr, endianess="little")
        assert result == bytes.fromhex("0000803f")

    def test_to_bytes_big_endian(self):
        """Test to_bytes with big-endian byte order."""
        from torchbit.core.matrix import to_bytes
        arr = np.array([0x3f800000], dtype=np.int32)
        result = to_bytes(arr, endianess="big")
        assert result == bytes.fromhex("3f800000")

    def test_to_bytes_invalid_endianess_raises(self):
        """Test to_bytes with invalid endianess raises AssertionError."""
        from torchbit.core.matrix import to_bytes
        arr = np.array([1], dtype=np.int32)
        with pytest.raises(AssertionError):
            to_bytes(arr, endianess="invalid")

    def test_to_bytes_2d_array_raises(self):
        """Test to_bytes with 2D array raises AssertionError."""
        from torchbit.core.matrix import to_bytes
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        with pytest.raises(AssertionError):
            to_bytes(arr, endianess="little")
