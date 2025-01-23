import torch
import numpy as np
from pathlib import Path

dtype_to_bits = {
    # for unisigned int, only support uint8, uint16 and uint32 are limited support
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
    torch.float64: 64,
}

standard_torch_dtype = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}
standard_numpy_dtype = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


def tensor2memhex(tensor: torch.Tensor, out_path: str | Path):
    # load a tensor and save as memhex that verilog could read
    # assert is a 2D tensor
    assert len(tensor.shape) == 2
    assert tensor.dtype in dtype_to_bits.keys()
    # get the bit length of dtype

    bit_length = dtype_to_bits[tensor.dtype]
    tensor = tensor.view(standard_torch_dtype[bit_length])
    tensor_numpy = tensor.numpy()

    with open(out_path, "w") as f:
        for tensor_row in tensor_numpy:
            tensor_row = np.flip(tensor_row, 0).copy()
            hex_row = tensor_row.tobytes().hex()
            f.write(hex_row + "\n")


def memhex2tensor(dtype: torch.dtype, in_path: str | Path):
    # read a memhex file and convert to tensor
    with open(in_path, "r") as f:
        lines = f.readlines()

    # get the bit length of dtype
    bit_length = dtype_to_bits[dtype]
    numpy_dtype = standard_numpy_dtype[bit_length]

    tensor_list = []
    for line in lines:
        byte_data = bytes.fromhex(line.strip())
        arr = np.frombuffer(byte_data, dtype=numpy_dtype)
        arr = np.flip(arr, 0).copy()
        tensor_row = torch.from_numpy(arr)
        tensor_list.append(tensor_row)

    tensor = torch.stack(tensor_list)
    return tensor.view(dtype)


