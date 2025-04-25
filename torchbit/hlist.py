import cocotb.binary
import torch
import numpy as np
import cocotb
from pathlib import Path
from .dtype import *
from .utils import *

def read_bigendian_arr(bytes, bit_length:int):
    # bigendian bytes -> numpy little end array
    numpy_dtype = standard_numpy_dtype[bit_length]
    be_numpy_dtype = standard_be_numpy_dtype[bit_length]
    arr = np.frombuffer(bytes, dtype=be_numpy_dtype)
    arr = arr.astype(numpy_dtype)
    arr = np.flip(arr, 0).copy() # decomple stride effet
    return arr

def to_bigendian_bytes(arr:np.ndarray):
    # numpy little end array -> bigendian bytes
    assert len(arr.shape) <= 1
    dtype = arr.dtype
    assert dtype in numpy_dtype_to_bits.keys(), f"{dtype} not in {numpy_dtype_to_bits.keys()}"

    bit_length = numpy_dtype_to_bits[dtype]
    be_numpy_dtype = standard_be_numpy_dtype[bit_length]
    arr = arr.copy()
    arr = np.flip(arr, 0).astype(be_numpy_dtype)
    return arr.tobytes()


class Hlist:
    # tensor -> memhex
    # tensor -> int
    # memhex -> tensor
    # int -> tensor
    def __init__(self, tensor: torch.Tensor = None):
        assert len(tensor.shape) == 2
        self.tensor = tensor

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        return Hlist(tensor)

    @staticmethod
    def from_memhexfile(in_path: str | Path, dtype: torch.dtype):
        # get the bit length of dtype
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        # read a memhex file and convert to tensor
        with open(in_path, "r") as f:
            lines = f.readlines()

        tensor_list = []
        for line in lines:
            byte_data = bytes.fromhex(line.strip())
            arr = read_bigendian_arr(byte_data, bit_length)
            tensor_row = torch.from_numpy(arr)
            tensor_list.append(tensor_row)

        tensor = torch.stack(tensor_list)
        return Hlist(tensor.view(dtype))

    @staticmethod
    def from_binfile(in_path: str | Path, num:int, dtype: torch.dtype):
        assert dtype in dtype_to_bits.keys()
        bit_length = dtype_to_bits[dtype]

        # read a binary file and convert to tensor
        with open(in_path, "rb") as f:  # 以二进制模式打开文件
            tensor_list = []
            while True:
                byte_data = f.read(bit_length * num // 8)
                if not byte_data:
                    break
                arr = read_bigendian_arr(byte_data, bit_length)
                tensor_row = torch.from_numpy(arr)
                tensor_list.append(tensor_row)

        tensor = torch.stack(tensor_list)
        return Hlist(tensor.view(dtype))

    def to_memhexfile(self, out_path: str | Path):
        # load a tensor and save as memhex that verilog could read
        # assert is a 2D tensor
        tensor = self.tensor

        assert len(tensor.shape) == 2
        assert tensor.dtype in dtype_to_bits.keys()
        # get the bit length of dtype

        bit_length = dtype_to_bits[tensor.dtype]
        tensor = tensor.view(standard_torch_dtype[bit_length])
        tensor_numpy = tensor.numpy()

        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for tensor_row in tensor_numpy:
                hex_row = to_bigendian_bytes(tensor_row).hex()
                f.write(hex_row + "\n")

    def to_binfile(self, out_path: str | Path):
        # load a tensor and save as binary that verilog could read
        # assert is a 2D tensor
        tensor = self.tensor

        assert len(tensor.shape) == 2
        assert tensor.dtype in dtype_to_bits.keys()
        # get the bit length of dtype

        bit_length = dtype_to_bits[tensor.dtype]
        tensor = tensor.view(standard_torch_dtype[bit_length])
        tensor_numpy = tensor.numpy()

        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:  # 修改为二进制写模式
            for tensor_row in tensor_numpy:
                byte_row = to_bigendian_bytes(tensor_row)
                f.write(byte_row)

    def to_tensor(self):
        return self.tensor
