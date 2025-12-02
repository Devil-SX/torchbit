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

if __name__ == "__main__":
    test_get_bit()
    test_get_bit_slice()
