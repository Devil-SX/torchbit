import torch
from ..core.dtype import *


def get_bit(value:int, num:int):
    assert isinstance(value, int) # Must use Python's built-in int type
    assert num >= 0
    return (value >> num) % 2


def get_slice_mask(high_close: int, low_close: int):
    num_bits = high_close - low_close + 1
    mask = ((1 << num_bits) - 1) << low_close
    return mask

def get_bit_slice(value: int, high_close: int, low_close: int):
    assert isinstance(value, int) # Must use Python's built-in int type
    mask = get_slice_mask(high_close, low_close)
    masked_value = value & mask
    return masked_value >> low_close

def replicate_bits(num: int, n: int) -> int:
    assert num > 0
    
    # Convert number to binary string, removing '0b' prefix
    binary = bin(n)[2:]
    
    # Replicate each bit num times
    replicated = ''.join(bit * num for bit in binary)
    
    # Convert back to integer
    return int(replicated, 2)


