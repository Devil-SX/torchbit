import torchbit.tools as struct_ops

def test_bit_field():
    print("Testing BitField...")
    # Test cases for BitField
    bf1 = struct_ops.BitField("field1", 4)
    bf1.set_value(0b1011)
    assert bf1.value == 0b1011, f"BitField set_value/value failed: Expected {0b1011}, Got {bf1.value}"
    bf1.set_value(0b11111) # Should truncate to 4 bits
    assert bf1.value == 0b1111, f"BitField truncation failed: Expected {0b1111}, Got {bf1.value}"
    print("BitField tests passed!")

def test_struct():
    print("\nTesting Struct...")

    # Create a Struct (Recommended instantiation)
    my_struct = struct_ops.Struct([
        struct_ops.BitField("a", 4),  # 4 bits
        struct_ops.BitField("b", 8),  # 8 bits
        struct_ops.BitField("c", 2)   # 2 bits
    ])

    # Test from_int and to_int
    test_value = 0b11_01010101_1111 # 2 (c) | 8 (b) | 4 (a) = 14 bits total
    my_struct.from_int(test_value)

    # Access fields via attributes
    assert my_struct.a.value == 0b1111, f"Struct from_int failed for field a: Expected {0b1111}, Got {my_struct.a.value}"
    assert my_struct.b.value == 0b01010101, f"Struct from_int failed for field b: Expected {0b01010101}, Got {my_struct.b.value}"
    assert my_struct.c.value == 0b11, f"Struct from_int failed for field c: Expected {0b11}, Got {my_struct.c.value}"

    # Modify a field via attribute and convert back to int
    my_struct.a.set_value(0b0010)
    reconstructed_value = my_struct.to_int()
    
    # Expected: c=0b11, b=0b01010101, a=0b0010
    # Value: 0b11_01010101_0010
    expected_reconstructed_value = (0b11 << (4 + 8)) | (0b01010101 << 4) | 0b0010
    assert reconstructed_value == expected_reconstructed_value, f"Struct to_int failed: Expected {bin(expected_reconstructed_value)}, Got {bin(reconstructed_value)}"
    
    # Test with a value larger than total_width
    large_value = 0b1_011_01010101_1111 # 15 bits total, struct is 14 bits
    my_struct.from_int(large_value)
    # The highest bit (15th) should be truncated
    assert my_struct.to_int() == test_value, f"Struct from_int with overflow failed: Expected {bin(test_value)}, Got {bin(my_struct.to_int())}"

    print("Struct tests passed!")

if __name__ == "__main__":
    test_bit_field()
    test_struct()
