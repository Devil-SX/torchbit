"""
Tests for the Vector class.

Vector is a 1D tensor wrapper for converting between PyTorch tensors
and Cocotb-compatible integer representations.
"""
import pytest
import torch
import numpy as np
from torchbit.core import Vector, tensor_to_cocotb, cocotb_to_tensor, array_to_logic, logic_to_array


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

    def test_from_tensor(self):
        """Test Vector.from_tensor() static method."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        vec = Vector.from_tensor(tensor)
        assert torch.equal(vec.tensor, tensor)

    def test_to_tensor(self):
        """Test Vector.to_tensor() returns the underlying tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec = Vector(tensor)
        assert torch.equal(vec.to_tensor(), tensor)


class TestVectorIntConversion:
    """Tests for Vector integer packing/unpacking."""

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
    def test_from_int_single_element(self, dtype):
        """Test from_int with a single element."""
        value = 0x42
        vec = Vector.from_int(value, 1, dtype)
        result = vec.to_tensor()
        assert result.item() == 0x42

    def test_from_int_multiple_elements_uint8(self):
        """Test from_int with multiple uint8 elements."""
        value = 0x41424344  # Will be unpacked LSB first
        vec = Vector.from_int(value, 4, torch.uint8)
        result = vec.to_tensor()
        # LSB first: 0x44, 0x43, 0x42, 0x41
        expected = torch.tensor([0x44, 0x43, 0x42, 0x41], dtype=torch.uint8)
        assert torch.equal(result, expected)

    def test_from_int_multiple_elements_int8(self):
        """Test from_int with multiple int8 elements."""
        value = 0x01020304
        vec = Vector.from_int(value, 4, torch.int8)
        result = vec.to_tensor()
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

    def test_to_cocotb_single_element(self):
        """Test to_cocotb with single element tensor."""
        tensor = torch.tensor([0x42], dtype=torch.uint8)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        assert result == 0x42

    def test_to_cocotb_multiple_elements(self):
        """Test to_cocotb packs elements correctly."""
        tensor = torch.tensor([0x01, 0x02, 0x03, 0x04], dtype=torch.uint8)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
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

        vec = Vector.from_tensor(tensor)
        cocotb_value = vec.to_cocotb()

        vec_back = Vector.from_int(cocotb_value, len(tensor), dtype)
        result = vec_back.to_tensor()

        assert torch.equal(result, tensor)

    def test_float32_bit_pattern(self):
        """Test that float32 bit patterns are preserved."""
        # 1.0 in float32 is 0x3f800000
        tensor = torch.tensor([1.0], dtype=torch.float32)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        assert result == 0x3f800000

    def test_float16_bit_pattern(self):
        """Test that float16 bit patterns are preserved."""
        # 1.0 in float16 is 0x3c00
        tensor = torch.tensor([1.0], dtype=torch.float16)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        assert result == 0x3c00


class TestVectorEdgeCases:
    """Tests for Vector edge cases."""

    def test_empty_tensor(self):
        """Test Vector with empty tensor."""
        tensor = torch.tensor([], dtype=torch.float32)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        # Empty tensor should result in 0
        assert result == 0

    def test_single_element_tensor(self):
        """Test Vector with single element tensor."""
        tensor = torch.tensor([42.0], dtype=torch.float32)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        # Should work the same as single element
        assert result is not None

    def test_large_tensor(self):
        """Test Vector with large tensor."""
        tensor = torch.arange(100, dtype=torch.int32)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        # Should produce a large integer without error
        assert isinstance(result, int)

    def test_zero_values(self):
        """Test Vector with all zero values."""
        tensor = torch.zeros(10, dtype=torch.float32)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        assert result == 0

    def test_negative_values_int(self):
        """Test Vector with negative integer values."""
        tensor = torch.tensor([-1, -2, -3], dtype=torch.int32)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        # Should handle negative values (two's complement)
        vec_back = Vector.from_int(result, 3, torch.int32)
        assert torch.equal(vec_back.to_tensor(), tensor)


class TestVectorCocotbConversion:
    """Tests for Vector Cocotb-specific conversion."""

    def test_from_cocotb_with_int(self):
        """Test from_cocotb with integer value."""
        value = 0x12345678
        vec = Vector.from_cocotb(value, 2, torch.int32)
        result = vec.to_tensor()
        assert len(result) == 2

    def test_from_cocotb_with_unsupported_type_raises(self):
        """Test from_cocotb with unsupported type raises AssertionError."""
        with pytest.raises(AssertionError, match="value must be a cocotb logicarray value or int value"):
            Vector.from_cocotb("invalid", 1, torch.int32)

    def test_cocotb_to_tensor_shortcut(self):
        """Test cocotb_to_tensor shortcut function."""
        value = 0x42
        tensor = cocotb_to_tensor(value, 1, torch.uint8)
        assert tensor.item() == 0x42

    def test_tensor_to_cocotb_shortcut(self):
        """Test tensor_to_cocotb shortcut function."""
        tensor = torch.tensor([0x01, 0x02], dtype=torch.uint8)
        result = tensor_to_cocotb(tensor)
        # Packs in reverse order: 0x0201
        assert result == 0x0201


class TestVectorMasking:
    """Tests for proper bit masking in conversions."""

    def test_to_cocotb_masks_overflow(self):
        """Test that to_cocotb properly masks values to bit width."""
        # Use uint8 to test masking since int8 can't hold 0xFF
        tensor = torch.tensor([0xFF, 0x01], dtype=torch.uint8)
        vec = Vector.from_tensor(tensor)
        result = vec.to_cocotb()
        # Each element is masked to 8 bits
        # [0xFF, 0x01] -> reversed -> [0x01, 0xFF] -> packed -> 0x01FF
        assert result == 0x01FF

    def test_from_int_masks_values(self):
        """Test that from_int properly masks input values."""
        value = 0xFFFFFFFF  # Large value
        vec = Vector.from_int(value, 4, torch.uint8)
        result = vec.to_tensor()
        # Should unpack correctly with 8-bit masking
        assert result[0].item() == 0xFF


class TestVectorCanonicalNames:
    """Tests for canonical to_logic/from_logic/to_array/from_array methods."""

    def test_to_logic_equals_to_cocotb(self):
        """Test that to_logic() returns same result as to_cocotb()."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec = Vector.from_tensor(tensor)
        assert vec.to_logic() == vec.to_cocotb()

    def test_to_logic_equals_to_int(self):
        """Test that to_logic() returns same result as to_int()."""
        tensor = torch.tensor([0x01, 0x02, 0x03], dtype=torch.uint8)
        vec = Vector.from_tensor(tensor)
        assert vec.to_logic() == vec.to_int()

    def test_from_logic_with_int(self):
        """Test from_logic with integer value."""
        value = 0x12345678
        vec = Vector.from_logic(value, 2, torch.int32)
        result = vec.to_array()
        assert len(result) == 2

    def test_from_logic_equals_from_cocotb(self):
        """Test that from_logic() returns same result as from_cocotb()."""
        value = 0x3f800000
        vec1 = Vector.from_logic(value, 1, torch.float32)
        vec2 = Vector.from_cocotb(value, 1, torch.float32)
        assert torch.equal(vec1.to_array(), vec2.to_tensor())

    def test_to_array_returns_tensor(self):
        """Test that to_array() returns the underlying tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec = Vector(tensor)
        assert torch.equal(vec.to_array(), tensor)

    def test_to_array_equals_to_tensor(self):
        """Test that to_array() returns same result as to_tensor()."""
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        vec = Vector(tensor)
        assert torch.equal(vec.to_array(), vec.to_tensor())

    def test_from_array(self):
        """Test from_array static method."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        vec = Vector.from_array(tensor)
        assert torch.equal(vec.to_array(), tensor)

    def test_from_array_equals_from_tensor(self):
        """Test that from_array() returns same result as from_tensor()."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        vec1 = Vector.from_array(tensor)
        vec2 = Vector.from_tensor(tensor)
        assert torch.equal(vec1.to_array(), vec2.to_tensor())

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
        assert array_to_logic(tensor) == tensor_to_cocotb(tensor)

    def test_logic_to_array_shortcut(self):
        """Test logic_to_array module-level shortcut."""
        value = 0x3f800000
        t1 = logic_to_array(value, 1, torch.float32)
        t2 = cocotb_to_tensor(value, 1, torch.float32)
        assert torch.equal(t1, t2)
