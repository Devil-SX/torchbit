"""
Data Conversion Example

Demonstrates Vector and Matrix classes for converting between PyTorch tensors
and hardware-compatible formats.

No DUT is required for this example - it runs as a Python script.
"""
import torch
from torchbit.core import Vector, VectorSequence, tensor_to_cocotb, cocotb_to_tensor, array_to_logic, logic_to_array


def test_vector_conversions():
    """Test Vector class for 1D tensor conversion."""
    print("=" * 60)
    print("Vector Conversion Tests")
    print("=" * 60)
    print()

    # Test 1: Float32 conversion
    print("Test 1: Float32 Vector Conversion")
    tensor_f32 = torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float32)
    vec = Vector.from_tensor(tensor_f32)
    cocotb_value = vec.to_cocotb()
    print(f"  Original tensor: {tensor_f32}")
    print(f"  Cocotb int value: 0x{cocotb_value:08x}")

    # Roundtrip
    vec_back = Vector.from_int(cocotb_value, len(tensor_f32), torch.float32)
    tensor_back = vec_back.to_tensor()
    print(f"  Roundtrip tensor: {tensor_back}")
    assert torch.allclose(tensor_f32, tensor_back), "Float32 roundtrip failed!"
    print("  PASSED: Float32 roundtrip")
    print()

    # Test 2: Int8 conversion
    print("Test 2: Int8 Vector Conversion")
    tensor_i8 = torch.tensor([10, 20, 30, 40], dtype=torch.int8)
    vec_i8 = Vector.from_tensor(tensor_i8)
    cocotb_i8 = vec_i8.to_cocotb()
    print(f"  Original tensor: {tensor_i8}")
    print(f"  Cocotb int value: 0x{cocotb_i8:08x}")

    vec_i8_back = Vector.from_int(cocotb_i8, len(tensor_i8), torch.int8)
    tensor_i8_back = vec_i8_back.to_tensor()
    print(f"  Roundtrip tensor: {tensor_i8_back}")
    assert torch.equal(tensor_i8, tensor_i8_back), "Int8 roundtrip failed!"
    print("  PASSED: Int8 roundtrip")
    print()

    # Test 3: Float16 conversion
    print("Test 3: Float16 Vector Conversion")
    tensor_f16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    vec_f16 = Vector.from_tensor(tensor_f16)
    cocotb_f16 = vec_f16.to_cocotb()
    print(f"  Original tensor: {tensor_f16}")
    print(f"  Cocotb int value: 0x{cocotb_f16:06x}")

    vec_f16_back = Vector.from_int(cocotb_f16, len(tensor_f16), torch.float16)
    tensor_f16_back = vec_f16_back.to_tensor()
    print(f"  Roundtrip tensor: {tensor_f16_back}")
    assert torch.allclose(tensor_f16, tensor_f16_back, atol=1e-3), "Float16 roundtrip failed!"
    print("  PASSED: Float16 roundtrip")
    print()

    # Test 4: BFloat16 conversion
    print("Test 4: BFloat16 Vector Conversion")
    tensor_bf16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    vec_bf16 = Vector.from_tensor(tensor_bf16)
    cocotb_bf16 = vec_bf16.to_cocotb()
    print(f"  Original tensor: {tensor_bf16}")
    print(f"  Cocotb int value: 0x{cocotb_bf16:06x}")

    vec_bf16_back = Vector.from_int(cocotb_bf16, len(tensor_bf16), torch.bfloat16)
    tensor_bf16_back = vec_bf16_back.to_tensor()
    print(f"  Roundtrip tensor: {tensor_bf16_back}")
    assert torch.allclose(tensor_bf16, tensor_bf16_back, atol=1e-2), "BFloat16 roundtrip failed!"
    print("  PASSED: BFloat16 roundtrip")
    print()

    # Test 5: Uint8 conversion
    print("Test 5: Uint8 Vector Conversion")
    tensor_u8 = torch.tensor([100, 150, 200, 255], dtype=torch.uint8)
    vec_u8 = Vector.from_tensor(tensor_u8)
    cocotb_u8 = vec_u8.to_cocotb()
    print(f"  Original tensor: {tensor_u8}")
    print(f"  Cocotb int value: 0x{cocotb_u8:08x}")

    vec_u8_back = Vector.from_int(cocotb_u8, len(tensor_u8), torch.uint8)
    tensor_u8_back = vec_u8_back.to_tensor()
    print(f"  Roundtrip tensor: {tensor_u8_back}")
    assert torch.equal(tensor_u8, tensor_u8_back), "Uint8 roundtrip failed!"
    print("  PASSED: Uint8 roundtrip")
    print()

    print("=" * 60)
    print("All Vector tests passed!")
    print("=" * 60)
    print()


def test_vector_sequence_conversions():
    """Test VectorSequence class for 2D tensor conversion with file I/O."""
    print("=" * 60)
    print("VectorSequence Conversion Tests")
    print("=" * 60)
    print()

    import tempfile
    from pathlib import Path

    # Test 1: Matrix to/from memory hex file
    print("Test 1: VectorSequence Memory Hex File Roundtrip")
    tensor_2d = torch.randn(4, 4, dtype=torch.float32)
    mat = VectorSequence.from_tensor(tensor_2d)
    print(f"  Original tensor shape: {tensor_2d.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        hex_path = Path(tmpdir) / "test.hex"
        mat.to_memhexfile(hex_path)
        print(f"  Written to: {hex_path}")

        mat_loaded = VectorSequence.from_memhexfile(hex_path, torch.float32)
        print(f"  Loaded tensor shape: {mat_loaded.tensor.shape}")

        assert torch.allclose(tensor_2d, mat_loaded.tensor), "VectorSequence hex roundtrip failed!"
        print("  PASSED: Memory hex file roundtrip")
    print()

    # Test 2: Matrix to/from binary file
    print("Test 2: VectorSequence Binary File Roundtrip")
    tensor_2d_int = torch.randint(-100, 100, (4, 8), dtype=torch.int32)
    mat_int = VectorSequence.from_tensor(tensor_2d_int)
    print(f"  Original tensor shape: {tensor_2d_int.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "test.bin"
        mat_int.to_binfile(bin_path)
        print(f"  Written to: {bin_path}")

        mat_int_loaded = VectorSequence.from_binfile(bin_path, num=8, dtype=torch.int32)
        print(f"  Loaded tensor shape: {mat_int_loaded.tensor.shape}")

        assert torch.equal(tensor_2d_int, mat_int_loaded.tensor), "VectorSequence binary roundtrip failed!"
        print("  PASSED: Binary file roundtrip")
    print()

    # Test 3: Matrix with different dtypes
    print("Test 3: VectorSequence with Different Dtypes")
    for dtype in [torch.int8, torch.int16, torch.float16]:
        if dtype == torch.int8:
            test_tensor = torch.randint(50, 100, (2, 4), dtype=dtype)
        elif dtype == torch.int16:
            test_tensor = torch.randint(100, 200, (2, 4), dtype=dtype)
        else:
            test_tensor = torch.randn(2, 4, dtype=dtype)

        test_mat = VectorSequence.from_tensor(test_tensor)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.bin"
            test_mat.to_binfile(test_path)

            if dtype == torch.float16:
                loaded_mat = VectorSequence.from_binfile(test_path, num=4, dtype=dtype)
            else:
                loaded_mat = VectorSequence.from_binfile(test_path, num=4, dtype=dtype)

            if dtype == torch.float16:
                assert torch.allclose(test_tensor, loaded_mat.tensor, atol=1e-3), f"{dtype} roundtrip failed!"
            else:
                assert torch.equal(test_tensor, loaded_mat.tensor), f"{dtype} roundtrip failed!"

            print(f"  PASSED: {dtype} roundtrip")
    print()

    print("=" * 60)
    print("All VectorSequence tests passed!")
    print("=" * 60)
    print()


def test_shortcut_functions():
    """Test shortcut functions tensor_to_cocotb and cocotb_to_tensor."""
    print("=" * 60)
    print("Shortcut Function Tests")
    print("=" * 60)
    print()

    # tensor_to_cocotb shortcut
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    cocotb_val = tensor_to_cocotb(tensor)
    print(f"tensor_to_cocotb({tensor.tolist()}) = 0x{cocotb_val:08x}")

    # cocotb_to_tensor shortcut
    tensor_back = cocotb_to_tensor(cocotb_val, len(tensor), torch.float32)
    print(f"cocotb_to_tensor(0x{cocotb_val:08x}, 3, float32) = {tensor_back.tolist()}")

    assert torch.allclose(tensor, tensor_back), "Shortcut roundtrip failed!"
    print("PASSED: Shortcut function roundtrip")
    print()

    print("=" * 60)
    print("Shortcut function tests passed!")
    print("=" * 60)
    print()


def main():
    """Run all data conversion tests."""
    print()
    print("*" * 60)
    print("TorchBit Data Conversion Example")
    print("*" * 60)
    print()
    print("This example demonstrates:")
    print("  - Vector class for 1D tensor <-> integer conversion")
    print("  - VectorSequence class for 2D tensor <-> file conversion")
    print("  - Support for multiple dtypes: float32, float16, bfloat16, int8, int16, int32, uint8")
    print("  - Roundtrip verification for all conversions")
    print()

    try:
        test_vector_conversions()
        test_vector_sequence_conversions()
        test_shortcut_functions()

        print()
        print("*" * 60)
        print("All tests passed successfully!")
        print("*" * 60)
        print()
        return 0
    except AssertionError as e:
        print()
        print(f"TEST FAILED: {e}")
        print()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
