import torchbit.utils.bit_ops as bit_utils

def test_get_bit():
    print("Testing get_bit...")
    # Test cases: (value, bit_pos, expected_bit)
    test_cases = [
        (0b1011, 0, 1),  # LSB
        (0b1011, 1, 1),
        (0b1011, 2, 0),
        (0b1011, 3, 1),  # MSB
        (0b0, 5, 0),     # Bit outside value range (should return 0)
        (0b11111111, 7, 1), # Max 8-bit
        (0b10000000, 7, 1)
    ]

    for value, bit_pos, expected_bit in test_cases:
        result = bit_utils.get_bit(value, bit_pos)
        assert result == expected_bit, f"get_bit({bin(value)}, {bit_pos}): Expected {expected_bit}, Got {result}"
        print(f"  Passed: get_bit({bin(value)}, {bit_pos}) == {result}")
    print("get_bit tests passed!")

def test_get_bit_slice():
    print("\nTesting get_bit_slice...")
    # Test cases: (value, high_close, low_close, expected_slice)
    test_cases = [
        (0b11011010, 5, 2, 0b0110), # bits 2 to 5 inclusive
        (0b10101010, 7, 0, 0b10101010), # full value
        (0b11110000, 3, 0, 0b0000), # lower 4 bits
        (0b11110000, 7, 4, 0b1111), # upper 4 bits
        (0b111, 2, 0, 0b111), # 3 bits
        (0b10000000000, 10, 10, 0b1) # single bit slice (bit 10)
    ]

    for value, high_close, low_close, expected_slice in test_cases:
        result = bit_utils.get_bit_slice(value, high_close, low_close)
        assert result == expected_slice, f"get_bit_slice({bin(value)}, {high_close}, {low_close}): Expected {bin(expected_slice)}, Got {bin(result)}"
        print(f"  Passed: get_bit_slice({bin(value)}, {high_close}, {low_close}) == {bin(result)}")
    print("get_bit_slice tests passed!")

def test_twos_complement():
    print("\nTesting twos_complement...")
    test_cases = [
        (-1, 8, 255),  # -1 in 8-bit
        (-2, 8, 254),  # -2 in 8-bit
        (-128, 8, 128), # Smallest 8-bit signed number
        (-1, 16, 65535), # -1 in 16-bit
        (-5, 4, 11),   # -5 in 4-bit (16 + (-5) = 11)
        (-7, 4, 9), # -7 in 4-bit (16 + (-7) = 9)
        (-8, 4, 8) # Smallest 4-bit signed number
    ]

    for value, width, expected_complement in test_cases:
        result = bit_utils.twos_complement(value, width)
        assert result == expected_complement, f"twos_complement({value}, {width}): Expected {expected_complement}, Got {result}"
        print(f"  Passed: twos_complement({value}, {width}) == {result}")
    
    # Test cases that should raise assertions (invalid inputs)
    print("  Testing assertion failures...")
    # Test case: positive value
    try:
        bit_utils.twos_complement(1, 8)
        assert False, "Should have raised assertion for positive value"
    except AssertionError as e:
        print(f"  Passed assertion for positive value: {e}")
    
    # Test case: value out of range (too negative)
    try:
        bit_utils.twos_complement(-9, 4) # min for 4-bit is -8
        assert False, "Should have raised assertion for value out of range"
    except AssertionError as e:
        print(f"  Passed assertion for value out of range: {e}")

    print("twos_complement tests passed!")


if __name__ == "__main__":
    test_get_bit()
    test_get_bit_slice()
    test_twos_complement()

