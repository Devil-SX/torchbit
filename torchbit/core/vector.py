import cocotb.binary
import torch
import numpy as np
import cocotb
from pathlib import Path
from .dtype import *
from ..utils.utils import *


class Vector:
    # tensor -> memhex
    # tensor -> int
    # memhex -> tensor
    # int -> tensor
    def __init__(self, tensor: torch.Tensor = None):
        assert len(tensor.shape) <= 1
        self.tensor = tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        return Vector(tensor)
    
    @staticmethod
    def from_int(value_int: int, num: int, dtype: torch.dtype):
        assert isinstance(value_int, int), "value must be an int, use Vector.from_int(value)"
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        vec = []
        mask = (1 << bit_length) - 1

        if num == 1:
            v = value_int & mask
            vec = np.array(v).astype(standard_numpy_dtype[bit_length])
            return Vector(torch.from_numpy(vec).view(dtype))
        else:
            for _ in range(num):
                vec.append(value_int & mask)
                value_int >>= bit_length
            vec = np.array(vec).astype(standard_numpy_dtype[bit_length])
            return Vector(torch.from_numpy(vec).view(dtype))

    @staticmethod
    def from_cocotb(value: cocotb.binary.BinaryValue | int, num: int, dtype: torch.dtype):
        assert isinstance(
            value, cocotb.binary.BinaryValue
        ) or isinstance(value, int), "value must be a cocotb binary value or int value, use Vector.from_cocotb(dut.io_xxx.value)"

        if isinstance(value, cocotb.binary.BinaryValue) and (("x" in value.binstr) or ("z" in value.binstr)):
            print("Warning: value is a X/Z value, use the zero result")
            return Vector(torch.zeros(num, dtype=dtype))

        value_int = value.integer if isinstance(value, cocotb.binary.BinaryValue) else value
        return Vector.from_int(value_int, num, dtype)

    def to_cocotb(self):
        assert len(self.tensor.shape) <= 1
        assert self.tensor.dtype in dtype_to_bits.keys()
        tensor = self.tensor

        bit_length = dtype_to_bits[tensor.dtype]
        tensor = tensor.view(standard_torch_dtype[bit_length])
        tensor_numpy = tensor.numpy()

        result = 0
        mask = (1 << bit_length) - 1
        if len(self.tensor.shape) == 1:
            for v in reversed(tensor_numpy.astype(object)):
                result = (result << bit_length) | (v & mask)
        else:
            result = tensor_numpy.item() & mask
        return result

    def to_tensor(self):
        return self.tensor


def tensor_to_cocotb(tensor:torch.Tensor):
    """
    Shortcut function to convert tensor to cocotb binary value
    """

    return Vector.from_tensor(tensor).to_cocotb()


def cocotb_to_tensor(value: cocotb.binary.BinaryValue, num: int, dtype: torch.dtype):
    """
    Shortcut function to convert cocotb binary value to tensor
    """
    return Vector.from_cocotb(value, num, dtype).to_tensor()