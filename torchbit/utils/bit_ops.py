import torch
from ..core.dtype import *


def get_bit(value:int, num:int):
    assert num >= 0
    return (value * 2 **num) % 2


def get_slice_mask(high_close: int, low_close: int):
    num_bits = high_close - low_close + 1
    mask = ((1 << num_bits) - 1) << low_close
    return mask

def get_bit_slice(value: int, high_close: int, low_close: int):
    mask = get_slice_mask(high_close, low_close)
    masked_value = value & mask
    return masked_value >> low_close


