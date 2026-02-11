"""
Tests for the Vector class.

Vector is a 1D tensor wrapper for converting between PyTorch tensors
and HDL-compatible integer representations.
"""
import pytest
import torch
import numpy as np
from torchbit.core import Vector, array_to_logic, logic_to_array


class TestVectorBasic:
    """Tests for basic Vector operations."""

    def test_vector_init_with_1d_tensor(self):
        """Test Vector initialization with a 1D tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec = Vector(tensor)
        assert torch.equal(vec.tensor, tensor)

    def test_vector_init_with_scalar_tensor(self):
        """Test Vector initialization with a scalar (0D) tensor."""
        tensor = torch.tensor(5.0, dtype=torch.float32)
        vec = Vector(tensor)
        assert vec.tensor.item() == 5.0

    def test_vector_init_with_2d_tensor_raises(self):
        """Test Vector initialization with 2D tensor raises AssertionError."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        with pytest.raises(AssertionError):
            Vector(tensor)

    def test_from_array(self):
        """Test Vector.from_array() static method."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        assert torch.equal(vec.tensor, tensor)

    def test_to_array(self):
        """Test Vector.to_array() returns the underlying tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec = Vector(tensor)
        assert torch.equal(vec.to_array(), tensor)


class TestVectorIntConversion:
    """Tests for Vector integer packing/unpacking."""

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
    def test_from_int_single_element(self, dtype):
        """Test from_int with a single element."""
        value = 0x42
        vec = Vector.from_int(value, 1, dtype)
        result = vec.to_array()
        assert result.item() == 0x42

    def test_from_int_multiple_elements_uint8(self):
        """Test from_int with multiple uint8 elements."""
        value = 0x41424344  # Will be unpacked LSB first
        vec = Vector.from_int(value, 4, torch.uint8)
        result = vec.to_array()
        # LSB first: 0x44, 0x43, 0x42, 0x41
        expected = torch.tensor([0x44, 0x43, 0x42, 0x41], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_from_int_multiple_elements_int8(self):
        """Test from_int with multiple int8 elements."""
        value = 0x01020304
        vec = Vector.from_int(value, 4, torch.int8)
        result = vec.to_array()
        expected = torch.tensor([4, 3, 2, 1], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_from_int_with_non_int_raises(self):
        """Test from_int with non-int value raises AssertionError."""
        with pytest.raises(AssertionError, match="value must be an int"):
            Vector.from_int("not an int", 1, torch.int32)

    def test_from_int_with_unsupported_dtype_raises(self):
        """Test from_int with unsupported dtype raises AssertionError."""
        with pytest.raises(AssertionError):
            Vector.from_int(0x42, 1, torch.bool)  # bool is not in dtype_to_bits

    def test_to_logic_single_element(self):
        """Test to_logic with single element tensor."""
        tensor = torch.tensor([0x42], dtype=torch.uint8)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        assert result == 0x42

    def test_to_logic_multiple_elements(self):
        """Test to_logic packs elements correctly."""
        tensor = torch.tensor([0x01, 0x02, 0x03, 0x04], dtype=torch.uint8)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        # Packs in reverse order (last element at MSB): 0x04030201
        assert result == 0x04030201


class TestVectorDtypes:
    """Tests for Vector with different data types."""

    @pytest.mark.parametrize("dtype,bit_length", [
        (torch.uint8, 8),
        (torch.int8, 8),
        (torch.int16, 16),
        (torch.int32, 32),
        (torch.int64, 64),
        (torch.float16, 16),
        (torch.bfloat16, 16),
        (torch.float32, 32),
        (torch.float64, 64),
    ])
    def test_roundtrip_dtype(self, dtype, bit_length):
        """Test roundtrip conversion for all supported dtypes."""
        if dtype == torch.uint8:
            tensor = torch.tensor([1, 2, 3, 4], dtype=dtype)
        elif dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            tensor = torch.tensor([1.5, 2.5, 3.5], dtype=dtype)
        elif dtype in [torch.int8, torch.int64]:
            # Use positive values for int8 and int64 to avoid sign extension issues
            tensor = torch.tensor([1, 2, 3, 4], dtype=dtype)
        else:
            # For int16 and int32, negative values work fine
            tensor = torch.tensor([100, -50, 200, -25], dtype=dtype)

        vec = Vector.from_array(tensor)
        logic_value = vec.to_logic()

        vec_back = Vector.from_int(logic_value, len(tensor), dtype)
        result = vec_back.to_array()

        assert torch.equal(result, tensor)

    def test_float32_bit_pattern(self):
        """Test that float32 bit patterns are preserved."""
        # 1.0 in float32 is 0x3f800000
        tensor = torch.tensor([1.0], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        assert result == 0x3f800000

    def test_float16_bit_pattern(self):
        """Test that float16 bit patterns are preserved."""
        # 1.0 in float16 is 0x3c00
        tensor = torch.tensor([1.0], dtype=torch.float16)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        assert result == 0x3c00


class TestVectorEdgeCases:
    """Tests for Vector edge cases."""

    def test_empty_tensor(self):
        """Test Vector with empty tensor."""
        tensor = torch.tensor([], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        # Empty tensor should result in 0
        assert result == 0

    def test_single_element_tensor(self):
        """Test Vector with single element tensor."""
        tensor = torch.tensor([42.0], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        # Should work the same as single element
        assert result is not None

    def test_large_tensor(self):
        """Test Vector with large tensor."""
        tensor = torch.arange(100, dtype=torch.int32)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        # Should produce a large integer without error
        assert isinstance(result, int)

    def test_zero_values(self):
        """Test Vector with all zero values."""
        tensor = torch.zeros(10, dtype=torch.float32)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        assert result == 0

    def test_negative_values_int(self):
        """Test Vector with negative integer values."""
        tensor = torch.tensor([-1, -2, -3], dtype=torch.int32)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        # Should handle negative values (two's complement)
        vec_back = Vector.from_int(result, 3, torch.int32)
        assert torch.equal(vec_back.to_array(), tensor)


class TestVectorLogicConversion:
    """Tests for Vector logic conversion."""

    def test_from_logic_with_int(self):
        """Test from_logic with integer value."""
        value = 0x12345678
        vec = Vector.from_logic(value, 2, torch.int32)
        result = vec.to_array()
        assert len(result) == 2

    def test_from_logic_with_unsupported_type_raises(self):
        """Test from_logic with unsupported type raises AssertionError."""
        with pytest.raises(AssertionError, match="value must be a cocotb logicarray value or int value"):
            Vector.from_logic("invalid", 1, torch.int32)

    def test_logic_to_array_shortcut(self):
        """Test logic_to_array shortcut function."""
        value = 0x42
        tensor = logic_to_array(value, 1, torch.uint8)
        assert tensor.item() == 0x42

    def test_array_to_logic_shortcut(self):
        """Test array_to_logic shortcut function."""
        tensor = torch.tensor([0x01, 0x02], dtype=torch.uint8)
        result = array_to_logic(tensor)
        # Packs in reverse order: 0x0201
        assert result == 0x0201


class TestVectorMasking:
    """Tests for proper bit masking in conversions."""

    def test_to_logic_masks_overflow(self):
        """Test that to_logic properly masks values to bit width."""
        # Use uint8 to test masking since int8 can't hold 0xFF
        tensor = torch.tensor([0xFF, 0x01], dtype=torch.uint8)
        vec = Vector.from_array(tensor)
        result = vec.to_logic()
        # Each element is masked to 8 bits
        # [0xFF, 0x01] -> reversed -> [0x01, 0xFF] -> packed -> 0x01FF
        assert result == 0x01FF

    def test_from_int_masks_values(self):
        """Test that from_int properly masks input values."""
        value = 0xFFFFFFFF  # Large value
        vec = Vector.from_int(value, 4, torch.uint8)
        result = vec.to_array()
        # Should unpack correctly with 8-bit masking
        assert result[0].item() == 0xFF


class TestVectorCanonicalNames:
    """Tests for canonical to_logic/from_logic/to_array/from_array methods."""

    def test_from_logic_with_int(self):
        """Test from_logic with integer value."""
        value = 0x12345678
        vec = Vector.from_logic(value, 2, torch.int32)
        result = vec.to_array()
        assert len(result) == 2

    def test_to_array_returns_tensor(self):
        """Test that to_array() returns the underlying tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec = Vector(tensor)
        assert torch.equal(vec.to_array(), tensor)

    def test_from_array(self):
        """Test from_array static method."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        assert torch.equal(vec.to_array(), tensor)

    def test_roundtrip_logic(self):
        """Test roundtrip with canonical from_array/to_logic/from_logic/to_array."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        logic_val = vec.to_logic()
        vec_back = Vector.from_logic(logic_val, len(tensor), torch.float32)
        assert torch.allclose(tensor, vec_back.to_array())

    def test_array_to_logic_shortcut(self):
        """Test array_to_logic module-level shortcut."""
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        expected = Vector.from_array(tensor).to_logic()
        assert array_to_logic(tensor) == expected

    def test_logic_to_array_shortcut(self):
        """Test logic_to_array module-level shortcut."""
        value = 0x3f800000
        t1 = logic_to_array(value, 1, torch.float32)
        expected = Vector.from_logic(value, 1, torch.float32).to_array()
        assert torch.equal(t1, expected)
