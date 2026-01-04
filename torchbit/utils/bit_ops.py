"""
Bit manipulation utilities for HDL value handling.

Provides functions for:
- Extracting individual bits
- Slicing bit ranges
- Sign conversion (signed/unsigned)
- Bit replication
- Two's complement calculation

Essential for low-level HDL signal processing and bit-field operations.
"""
import torch
from ..core.dtype import *


def get_bit(value: int, num: int) -> int:
    """Extract a single bit from a value.

    Args:
        value: Integer to extract from.
        num: Bit position (0 = LSB, higher values = more significant).

    Returns:
        0 or 1 at the specified bit position.

    Example:
        >>> get_bit(0b1010, 0)
        0
        >>> get_bit(0b1010, 1)
        1
        >>> get_bit(0b1010, 3)
        1
    """
    assert isinstance(value, int)  # Must use Python's built-in int type
    assert num >= 0
    return (value >> num) % 2


def get_slice_mask(high_close: int, low_close: int) -> int:
    """Create a mask for a bit range.

    Generates a bitmask with 1s from low_close to high_close (inclusive)
    and 0s elsewhere.

    Args:
        high_close: High bit position (inclusive).
        low_close: Low bit position (inclusive).

    Returns:
        Integer mask with 1s in the specified range.

    Example:
        >>> get_slice_mask(5, 3)  # Mask bits 3, 4, 5
        56  # 0b111000
        >>> hex(get_slice_mask(7, 0))  # All 8 bits
        '0xff'
    """
    num_bits = high_close - low_close + 1
    mask = ((1 << num_bits) - 1) << low_close
    return mask


def get_bit_slice(value: int, high_close: int, low_close: int) -> int:
    """Extract a range of bits from a value.

    Args:
        value: Integer to extract from.
        high_close: High bit position (inclusive).
        low_close: Low bit position (inclusive).

    Returns:
        Integer value of the extracted bits, right-aligned.

    Example:
        >>> get_bit_slice(0b10101010, 5, 2)  # Extract bits 2-5
        10  # 0b1010
        >>> get_bit_slice(0xDEADBEEF, 7, 0)  # Extract lower byte
        239  # 0xEF
    """
    assert isinstance(value, int)  # Must use Python's built-in int type
    mask = get_slice_mask(high_close, low_close)
    masked_value = value & mask
    return masked_value >> low_close


def signed(value: int, width: int) -> int:
    """Convert an unsigned integer to a signed integer.

    Interprets the value as a signed integer of the specified width
    using two's complement representation.

    Args:
        value: Unsigned integer value.
        width: Bit width for interpretation (must be positive).

    Returns:
        Signed integer value. Negative if the sign bit is set.

    Example:
        >>> signed(0xFF, 8)
        -1
        >>> signed(0x7F, 8)
        127
        >>> signed(0x80, 8)
        -128
    """
    assert isinstance(value, int)
    assert isinstance(width, int) and width > 0
    sign_bit = 1 << (width - 1)
    if value & sign_bit:
        # Negative number
        return value - (1 << width)
    else:
        # Positive number
        return value


def unsigned(value: int, width: int) -> int:
    """Convert an integer to unsigned with bit masking.

    Masks the value to the specified bit width, ensuring it fits
    within the unsigned range for that width.

    Args:
        value: Integer value to mask.
        width: Bit width to mask to (must be positive).

    Returns:
        Unsigned integer within the specified width.

    Example:
        >>> unsigned(-1, 8)
        255
        >>> unsigned(0x100, 8)
        0
    """
    assert isinstance(value, int)
    assert isinstance(width, int) and width > 0
    return value & ((1 << width) - 1)


def replicate_bits(num: int, n: int) -> int:
    """Replicate each bit in n, num times.

    Takes each bit in the binary representation of n and replicates
    it the specified number of times. Useful for creating write masks
    or expanding control signals.

    Args:
        num: Number of times to replicate each bit.
        n: Value to replicate.

    Returns:
        Bit-replicated value.

    Example:
        >>> replicate_bits(4, 0b1011)
        4369  # 0b1000100010001
        >>> replicate_bits(8, 0x1)
        255  # 0xFF (8-bit mask)
        >>> replicate_bits(4, 0x3)
        0x3333
    """
    assert num > 0

    # Convert number to binary string, removing '0b' prefix
    binary = bin(n)[2:]

    # Replicate each bit num times
    replicated = ''.join(bit * num for bit in binary)

    # Convert back to integer
    return int(replicated, 2)


def twos_complement(value: int, width: int) -> int:
    """Calculate two's complement representation of a negative integer.

    Args:
        value: Negative integer to convert (must be negative).
        width: Bit width for the representation (must be positive).

    Returns:
        Two's complement bit pattern as a positive integer.

    Raises:
        AssertionError: If value is positive or out of range for the width.

    Example:
        >>> twos_complement(-1, 8)
        255  # 0xFF
        >>> twos_complement(-128, 8)
        128  # 0x80
        >>> twos_complement(-1, 16)
        65535  # 0xFFFF
    """
    assert isinstance(value, int)
    assert isinstance(width, int) and width > 0
    assert value < 0, "Value must be negative for two's complement calculation"
    # Ensure the absolute value can be represented within width-1 bits (for signed representation)
    # The smallest negative number representable with `width` bits is -(1 << (width - 1))
    assert abs(value) <= (1 << (width - 1)), f"Value {value} is out of range for {width} bits signed representation"

    # Two's complement for a negative number is (1 << width) + value
    return (1 << width) + value
