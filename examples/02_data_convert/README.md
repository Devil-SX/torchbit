# Data Conversion Example

This example demonstrates the Vector and Matrix classes for converting between PyTorch tensors and hardware-compatible formats.

## Overview

- **Vector**: 1D tensor ↔ integer conversion for HDL interfaces
- **Matrix**: 2D tensor ↔ file conversion (hex/bin) for memory initialization
- **Purpose**: Show how to move data between PyTorch and hardware simulations

## Key Classes

### Vector
For 1D tensors (arrays, vectors):
- `Vector.from_tensor(tensor)` - Create Vector from tensor
- `vec.to_cocotb()` - Convert to integer for HDL
- `Vector.from_int(value, num, dtype)` - Create from integer
- `vec.to_tensor()` - Get back tensor

### Matrix
For 2D tensors (matrices, memory):
- `Matrix.from_tensor(tensor)` - Create Matrix from tensor
- `mat.to_memhexfile(path)` - Write to memory hex file
- `Matrix.from_memhexfile(path, dtype)` - Read from hex file
- `mat.to_binfile(path)` - Write to binary file
- `Matrix.from_binfile(path, num, dtype)` - Read from binary file

## Supported Data Types

- Floating-point: `float16`, `bfloat16`, `float32`, `float64`
- Integer: `int8`, `int16`, `int32`, `int64`, `uint8`

## Running the Example

```bash
cd examples/02_data_convert
python run.py
```

Or run the test file directly:

```bash
python test_data_convert.py
```

## Example Output

```
Vector Conversion Tests
========================
Test 1: Float32 Vector Conversion
  Original tensor: tensor([1.5000, 2.5000, 3.5000, 4.5000])
  Cocotb int value: 0x40600000
  Roundtrip tensor: tensor([1.5000, 2.5000, 3.5000, 4.5000])
  PASSED: Float32 roundtrip
...
All tests passed successfully!
```

## File Structure

```
02_data_convert/
├── README.md             # This file
├── run.py                # Entry point script
└── test_data_convert.py  # Conversion tests and demonstrations
```

## Key Concepts Demonstrated

1. **Vector Class**: Converting 1D tensors to packed integers
2. **Matrix Class**: Converting 2D tensors to/from files
3. **Roundtrip Conversion**: Verifying data integrity
4. **Multiple Dtypes**: Support for various data types
5. **Endianness**: Little-endian and big-endian byte ordering
