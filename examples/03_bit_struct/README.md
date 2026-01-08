# BitStruct Usage Example

This example demonstrates the BitField and BitStruct classes for bit-level data manipulation, useful for working with hardware registers and packed data structures.

## Overview

- **BitField**: Represents a named bit field within a structure
- **BitStruct**: Factory function that creates packed data structure classes
- **Purpose**: Model hardware registers, instruction formats, protocol headers

## Key Classes

### BitField
```python
field = BitField("name", width)  # Create a bit field
field.set_value(value)            # Set field value
field.value                       # Get field value
```

### BitStruct
```python
# Define a structure
MyStruct = BitStruct(
    "MyStruct",
    [
        BitField("field1", 8),
        BitField("field2", 8),
        BitField("field3", 16),
    ]
)

# Create instance from integer
inst = MyStruct.from_int(0xAABBCCDD)

# Access fields
print(inst.field1)  # 0xDD
print(inst.field2)  # 0xCC
print(inst.field3)  # 0xBBAA

# Modify fields
inst.field1 = 0xFF

# Convert back to integer
value = inst.to_int()
```

### Comparison
```python
from torchbit.tools.bit_struct import compare_bit_struct

result = compare_bit_struct(inst1, inst2)
# Returns None if identical, otherwise returns mismatched field name
```

## Running the Example

```bash
cd examples/03_bit_struct
python run.py
```

## Use Cases

1. **CPU Registers**: Model status, control, and configuration registers
2. **Instruction Formats**: Decode/encode CPU instructions
3. **Protocol Headers**: Network packet headers, bus protocols
4. **Memory-Mapped I/O**: GPIO, UART, SPI configuration registers
5. **Packed Data**: Any bit-level data structure

## Example: RISC-V Instruction

```python
InstructionStruct = BitStruct(
    "InstructionStruct",
    [
        BitField("opcode", 7),
        BitField("rd", 5),
        BitField("rs1", 5),
        BitField("rs2", 5),
        BitField("funct3", 3),
        BitField("funct7", 7),
    ]
)

inst = InstructionStruct.from_int(0x00F50513)  # addi x10, x10, 0
print(f"opcode: {inst.opjective}")
print(f"rd: x{inst.rd}")
print(f"rs1: x{inst.rs1}")
```

## File Structure

```
03_bit_struct/
├── README.md            # This file
├── run.py               # Entry point script
└── test_bit_struct.py   # BitStruct demonstrations
```

## Key Concepts Demonstrated

1. **BitField Creation**: Named bit fields with specified width
2. **BitStruct Definition**: Groups of bit fields
3. **Roundtrip Conversion**: Integer ↔ Struct
4. **Field Ordering**: LSB-first packing
5. **Field Access**: Direct attribute access
6. **Comparison**: Structure comparison utilities
7. **Overflow Handling**: Automatic truncation to field width
