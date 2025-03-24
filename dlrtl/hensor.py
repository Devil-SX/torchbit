import cocotb.binary
import torch
import numpy as np
import cocotb
from pathlib import Path
from .dtype import *
from .utils import *


class Hensor:
    # tensor -> memhex
    # tensor -> int
    # memhex -> tensor
    # int -> tensor
    def __init__(self, tensor: torch.Tensor = None):
        assert len(self.tensor.shape) <= 1
        self.tensor = tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        return Hensor(tensor)

    def from_cocotb(value: cocotb.binary.BinaryValue, num: int, dtype: torch.dtype):
        assert isinstance(
            value, cocotb.binary.BinaryValue
        ), "value must be a cocotb binary value, use Hensor.from_cocotb(dut.io_xxx.value)"
        if ("x" in value.binstr) or ("z" in value.binstr):
            print("Warning: value is a X/Z value, use the zero result")
            return Hensor(torch.zeros(num, dtype=dtype))

        value_int = value.integer
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        vec = []
        mask = (1 << bit_length) - 1

        if num == 1:
            v = value_int & mask
            vec = np.array(v).astype(standard_numpy_dtype[bit_length])
            return Hensor(torch.from_numpy(vec).view(dtype))
        else:
            for _ in range(num):
                vec.append(value_int & mask)
                value_int >>= bit_length
            vec = np.array(vec).astype(standard_numpy_dtype[bit_length])
            return Hensor(torch.from_numpy(vec).view(dtype))

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
