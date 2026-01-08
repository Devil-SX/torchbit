"""
BitStruct Usage Example

Demonstrates BitField and BitStruct classes for bit-level data manipulation.
Useful for working with hardware registers, packed data structures, and protocol fields.
"""
import torch
from torchbit.tools.bit_struct import BitStruct, BitField, compare_bit_struct
from torchbit.utils.bit_ops import get_bit, get_bit_slice, twos_complement


def test_bit_field_operations():
    """Test BitField creation and manipulation."""
    print("=" * 60)
    print("BitField Operations")
    print("=" * 60)
    print()

    # Create a simple bit field
    field = BitField("opcode", 4)
    print(f"Created BitField: name={field.name}, width={field.width}")

    # Set and get values
    field.set_value(0xA)
    print(f"Set value to 0x{field.value:X}")
    print(f"Get value: {field.value}")
    print()

    # Test overflow handling
    print("Testing overflow (4-bit field, value 0x123):")
    field.set_value(0x123)
    print(f"  Stored value: 0x{field.value:X} (truncated to 4 bits)")
    print()


def test_bit_struct_creation():
    """Test BitStruct creation with various field configurations."""
    print("=" * 60)
    print("BitStruct Creation")
    print("=" * 60)
    print()

    # Example 1: Instruction format (LSB-first)
    print("Example 1: RISC-V Instruction Format (32-bit)")
    print("  Fields: opcode(7), rd(5), rs1(5), rs2(5), funct3(3), funct7(7)")
    print()

    InstructionStruct = BitStruct(
        "InstructionStruct",
        [
            BitField("opcode", 7),
            BitField("rd", 5),
            BitField("rs1", 5),
            BitField("rs2", 5),
            BitField("funct3", 3),
            BitField("funct7", 7),
        ],
    )

    # Create an instance from integer
    inst_value = 0x00F50513  # addi x10, x10, 0
    inst = InstructionStruct.from_int(inst_value)
    print(f"  Instruction: 0x{inst_value:08X}")
    print(f"  opcode: 0x{inst.opjective:X}")
    print(f"  rd: x{inst.rd}")
    print(f"  rs1: x{inst.rs1}")
    print(f"  rs2: x{inst.rs2}")
    print(f"  funct3: 0x{inst.funct3:X}")
    print(f"  funct7: 0x{inst.funct7:X}")
    print()

    # Roundtrip: back to integer
    inst_back = inst.to_int()
    print(f"  Roundtrip: 0x{inst_back:08X}")
    assert inst_value == inst_back, "Roundtrip failed!"
    print("  PASSED: Roundtrip")
    print()

    # Example 2: Control register
    print("Example 2: Control Register (32-bit)")
    print("  Fields: enable(1), mode(2), reserved(5), count(16), parity(1), error(1)")
    print()

    ControlStruct = BitStruct(
        "ControlStruct",
        [
            BitField("enable", 1),
            BitField("mode", 2),
            BitField("reserved", 5),
            BitField("count", 16),
            BitField("parity", 1),
            BitField("error", 1),
        ],
    )

    ctrl = ControlStruct.from_int(0x00008021)
    print(f"  Register value: 0x{ctrl.to_int():08X}")
    print(f"  enable: {ctrl.enable}")
    print(f"  mode: {ctrl.mode}")
    print(f"  reserved: 0x{ctrl.reserved:02X}")
    print(f"  count: {ctrl.count}")
    print(f"  parity: {ctrl.parity}")
    print(f"  error: {ctrl.error}")
    print()

    # Modify fields using setattr
    ctrl.enable = 1
    ctrl.mode = 2
    ctrl.count = 1000
    print(f"  After modification: 0x{ctrl.to_int():08X}")
    print(f"  New count: {ctrl.count}")
    print()


def test_bit_struct_field_ordering():
    """Test BitStruct with different field orderings."""
    print("=" * 60)
    print("BitStruct Field Ordering")
    print("=" * 60)
    print()

    # Test LSB-first ordering (default)
    print("LSB-first ordering (fields start from bit 0):")
    LSBStruct = BitStruct(
        "LSBStruct",
        [
            BitField("field_a", 4),  # bits 3:0
            BitField("field_b", 4),  # bits 7:4
            BitField("field_c", 8),  # bits 15:8
        ],
    )

    lsb_inst = LSBStruct.from_int(0xAABBCCDD & 0xFFFF)
    print(f"  Value: 0x{lsb_inst.to_int():04X}")
    print(f"  field_a (bits 3:0): 0x{lsb_inst.field_a:X}")
    print(f"  field_b (bits 7:4): 0x{lsb_inst.field_b:X}")
    print(f"  field_c (bits 15:8): 0x{lsb_inst.field_c:X}")
    print()

    # Demonstrate packing order
    print("Understanding packing order:")
    print("  When packing 0xAABBCCDD with fields [4, 4, 8] bits:")
    print("  - field_a gets lowest 4 bits: 0xD")
    print("  - field_b gets next 4 bits: 0xC")
    print("  - field_c gets next 8 bits: 0xBB")
    print()


def test_bit_struct_inspect():
    """Test BitStruct inspect functionality."""
    print("=" * 60)
    print("BitStruct Inspect")
    print("=" * 60)
    print()

    TestStruct = BitStruct(
        "TestStruct",
        [
            BitField("a", 4),
            BitField("b", 8),
            BitField("c", 4),
        ],
    )

    inst = TestStruct.from_int(0xABC)

    print("Inspecting BitStruct instance:")
    inst.inspect()
    print()


def test_bit_struct_comparison():
    """Test BitStruct comparison utilities."""
    print("=" * 60)
    print("BitStruct Comparison")
    print("=" * 60)
    print()

    TestStruct = BitStruct(
        "TestStruct",
        [
            BitField("field1", 8),
            BitField("field2", 8),
            BitField("field3", 8),
        ],
    )

    # Create two instances
    inst1 = TestStruct.from_int(0x010203)
    inst2 = TestStruct.from_int(0x010203)
    inst3 = TestStruct.from_int(0x010204)

    print("Comparing inst1 (0x010203) and inst2 (0x010203):")
    result = compare_bit_struct(inst1, inst2)
    if result is None:
        print("  No mismatches - structures are identical")
    print()

    print("Comparing inst1 (0x010203) and inst3 (0x010204):")
    result = compare_bit_struct(inst1, inst3)
    if result:
        print(f"  Mismatch found in field: {result}")
    print()

    # Test type mismatch
    OtherStruct = BitStruct(
        "OtherStruct",
        [
            BitField("x", 8),
            BitField("y", 8),
        ],
    )

    other = OtherStruct.from_int(0x0102)
    print("Comparing inst1 with different struct type:")
    result = compare_bit_struct(inst1, other)
    if result is not None:
        print(f"  Result: {result}")
    print()


def test_bit_operations():
    """Test basic bit operation utilities."""
    print("=" * 60)
    print("Bit Operation Utilities")
    print("=" * 60)
    print()

    # get_bit
    value = 0b10101010
    print(f"Value: 0b{value:08b} (0x{value:02X})")
    for i in range(8):
        bit_val = get_bit(value, i)
        print(f"  get_bit(0x{value:02X}, {i}) = {bit_val}")
    print()

    # get_bit_slice
    value2 = 0x12345678
    print(f"Value: 0x{value2:08X}")
    slice_val = get_bit_slice(value2, 8, 16)
    print(f"  get_bit_slice(0x{value2:08X}, 8, 16) = 0x{slice_val:04X}")
    print()

    # twos_complement
    print("Two's complement:")
    for val in [127, 128, 255, 256]:
        tc = twos_complement(val, 8)
        print(f"  twos_complement({val}, 8) = {tc}")
    print()


def test_practical_examples():
    """Demonstrate practical use cases."""
    print("=" * 60)
    print("Practical Examples")
    print("=" * 60)
    print()

    # Example 1: CPU Status Register
    print("Example 1: CPU Status Register")
    StatusStruct = BitStruct(
        "StatusStruct",
        [
            BitField("carry", 1),
            BitField("overflow", 1),
            BitField("zero", 1),
            BitField("negative", 1),
            BitField("irq_enable", 1),
            BitField("fiq_enable", 1),
            BitField("mode", 5),
            BitField("reserved", 22),
        ],
    )

    status = StatusStruct.from_int(0x00000030)
    print(f"  Status: 0x{status.to_int():08X}")
    print(f"  Zero flag: {status.zero}")
    print(f"  IRQ enabled: {status.irq_enable}")
    print(f"  Mode: {status.mode}")
    print()

    # Example 2: Memory-mapped I/O register
    print("Example 2: GPIO Direction Register (16 bits)")
    GPIOStruct = BitStruct(
        "GPIOStruct",
        [BitField(f"pin{i}", 1) for i in range(16)]
    )

    # Set pins 0-7 as output, 8-15 as input
    gpio_value = 0x00FF
    gpio = GPIOStruct.from_int(gpio_value)
    print(f"  GPIO direction: 0x{gpio.to_int():04X}")
    print("  Pin directions (0=input, 1=output):")
    for i in range(16):
        direction = getattr(gpio, f"pin{i}")
        print(f"    Pin {i}: {'OUT' if direction else 'IN'}")
    print()


def main():
    """Run all BitStruct examples."""
    print()
    print("*" * 60)
    print("TorchBit BitStruct Usage Example")
    print("*" * 60)
    print()
    print("This example demonstrates:")
    print("  - BitField: Individual bit fields")
    print("  - BitStruct: Packed data structures")
    print("  - Roundtrip conversion: struct ↔ integer")
    print("  - Field ordering and bit positioning")
    print("  - Comparison utilities")
    print("  - Bit manipulation helpers")
    print()

    try:
        test_bit_field_operations()
        test_bit_struct_creation()
        test_bit_struct_field_ordering()
        test_bit_struct_inspect()
        test_bit_struct_comparison()
        test_bit_operations()
        test_practical_examples()

        print()
        print("*" * 60)
        print("All examples completed successfully!")
        print("*" * 60)
        print()
        return 0
    except AssertionError as e:
        print()
        print(f"ERROR: {e}")
        print()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
