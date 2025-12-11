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

def test_bit_struct():
    print("\nTesting BitStruct (LSB First)...")

    # Create a BitStruct (Recommended instantiation)
    MyStruct = struct_ops.BitStruct(fields=[
        struct_ops.BitField("a", 4),  # 4 bits (LSB)
        struct_ops.BitField("b", 8),  # 8 bits
        struct_ops.BitField("c", 2)   # 2 bits (MSB)
    ], lsb_first=True)

    my_struct = MyStruct()

    # Test from_int and to_int
    test_value = 0b11_01010101_1111 # 2 (c) | 8 (b) | 4 (a) = 14 bits total
    my_struct.from_int(test_value)

    # Access fields via attributes
    assert my_struct.a.value == 0b1111, f"BitStruct from_int failed for field a: Expected {0b1111}, Got {my_struct.a.value}"
    assert my_struct.b.value == 0b01010101, f"BitStruct from_int failed for field b: Expected {0b01010101}, Got {my_struct.b.value}"
    assert my_struct.c.value == 0b11, f"BitStruct from_int failed for field c: Expected {0b11}, Got {my_struct.c.value}"

    # Modify a field via attribute and convert back to int
    my_struct.a.set_value(0b0010)
    reconstructed_value = my_struct.to_int()
    
    # Expected: c=0b11, b=0b01010101, a=0b0010
    # Value: 0b11_01010101_0010
    expected_reconstructed_value = (0b11 << (4 + 8)) | (0b01010101 << 4) | 0b0010
    assert reconstructed_value == expected_reconstructed_value, f"BitStruct to_int failed: Expected {bin(expected_reconstructed_value)}, Got {bin(reconstructed_value)}"
    
    # Test with a value larger than total_width
    large_value = 0b1_011_01010101_1111 # 15 bits total, struct is 14 bits
    my_struct.from_int(large_value)
    # The highest bit (15th) should be truncated
    assert my_struct.to_int() == test_value, f"BitStruct from_int with overflow failed: Expected {bin(test_value)}, Got {bin(my_struct.to_int())}"

    print("BitStruct tests (LSB First) passed!")
    my_struct.inspect()

    # Test __setattr__ functionality
    print("\nTesting BitStruct __setattr__...")
    my_struct.a = 0b0101 # Set 'a' directly
    assert my_struct.a.value == 0b0101, f"BitStruct __setattr__ failed: Expected {0b0101}, Got {my_struct.a.value}"
    # Verify to_int still works after direct assignment
    new_reconstructed_value = my_struct.to_int()
    # Expected: c=0b11, b=0b01010101, a=0b0101
    expected_new_reconstructed_value = (0b11 << (4 + 8)) | (0b01010101 << 4) | 0b0101
    assert new_reconstructed_value == expected_new_reconstructed_value, f"BitStruct to_int after __setattr__ failed: Expected {bin(expected_new_reconstructed_value)}, Got {bin(new_reconstructed_value)}"
    print("BitStruct __setattr__ test passed!")

    print("\nTesting BitStruct (MSB First)...")
    # Create a BitStruct (MSB First)
    # List order: a, b, c. 
    # If MSB first: a is MSB, c is LSB.
    # Value layout: [a (4)] [b (8)] [c (2)]
    MsbStruct = struct_ops.BitStruct(fields=[
        struct_ops.BitField("a", 4),  
        struct_ops.BitField("b", 8),  
        struct_ops.BitField("c", 2)   
    ], lsb_first=False)

    msb_struct = MsbStruct()

    # Construct value: a=0b1111, b=0b01010101, c=0b11
    # Binary: 1111 01010101 11
    # Hex: 0xF 0x55 0x3 -> F553 (but bits are continuous)
    # Value: (15 << 10) | (85 << 2) | 3
    # 15 * 1024 + 85 * 4 + 3 = 15360 + 340 + 3 = 15703
    msb_val = (0b1111 << 10) | (0b01010101 << 2) | 0b11
    
    msb_struct.from_int(msb_val)
    
    assert msb_struct.a.value == 0b1111, f"MSB BitStruct a failed: Got {bin(msb_struct.a.value)}"
    assert msb_struct.b.value == 0b01010101, f"MSB BitStruct b failed: Got {bin(msb_struct.b.value)}"
    assert msb_struct.c.value == 0b11, f"MSB BitStruct c failed: Got {bin(msb_struct.c.value)}"
    
    assert msb_struct.to_int() == msb_val, "MSB BitStruct to_int failed"
    print("BitStruct tests (MSB First) passed!")
    msb_struct.inspect()

if __name__ == "__main__":
    test_bit_field()
    test_bit_struct()
