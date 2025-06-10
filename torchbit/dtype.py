import torch
import numpy as np

dtype_to_bits = {
    # for unisigned int, only support uint8. uint16 and uint32 are limited support
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

standard_torch_dtype = {
    8: torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}

numpy_dtype_to_bits = {
    np.int8:8,
    np.int16:16,
    np.int32:32,
    np.int64:64,
    np.dtype("int8") : 8,
    np.dtype("int16") : 16,
    np.dtype("int32") : 32,
    np.dtype("int64") : 64,
}

standard_numpy_dtype = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
standard_be_numpy_dtype = {8: ">i1", 16: ">i2", 32: ">i4", 64: ">i8"}
standard_le_numpy_dtype = {8: "<i1", 16: "<i2", 32: "<i4", 64: "<i8"}
