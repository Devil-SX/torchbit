from pathlib import Path
from os import PathLike
import torch
import numpy as np
import einops
from ..core.vector import Vector
from ..debug.judge import compare
from dataclasses import dataclass

# TestVector -> Sequential
# TestVector -> (addr, Sequential)


class AddressMapping:
    def __init__(self, base, strides: tuple, max_values: tuple):
        self.strides = strides
        self.max_values = max_values
        self.base = base

    def get_addr_list(self):
        indexs = list(np.ndindex(self.max_values))
        indexs = [np.array(index) for index in indexs]
        indexs = np.stack(indexs)  # [N, M]
        strides = np.array(self.strides)  # [M]
        addrs = einops.reduce(indexs * strides, "n m -> n", "sum") + self.base  # [N]
        return addrs


class TileMapping:
    dtype: torch.dtype
    sw_einops: str
    hw_einops: str
    hw_temp_dim: dict
    hw_spat_dim: dict
    base_addr: int = 0
    strides: dict = None

    def __post_init__(self):
        assert all(isinstance(k, str) for k in self.hw_temp_dim.keys()) and all(
            isinstance(v, int) for v in self.hw_temp_dim.values()
        ), "hw_temp_dim must be a dict of str to int"
        assert all(isinstance(k, str) for k in self.hw_spat_dim.keys()) and all(
            isinstance(v, int) for v in self.hw_spat_dim.values()
        ), "hw_spat_dim must be a dict of str to int"

        self.num = int(np.prod(list(self.hw_spat_dim.values())))
        self.sw_to_hw_formula = f"{self.sw_einops} -> {self.hw_einops}"
        self.hw_to_sw_formula = f"{self.hw_einops} -> {self.sw_einops}"

        if self.strides is not None:
            self.address_mapping = AddressMapping(
                self.base_addr,
                tuple(self.strides.values()),
                tuple(self.hw_temp_dim.values()),
            )


def tensor_to_cocotb_seq(tensor: torch.Tensor, mapping: TileMapping):
    tensor_seq = einops.rearrange(
        tensor, mapping.sw_to_hw_formula, **mapping.hw_temp_dim, **mapping.hw_spat_dim
    )  # [N, M]
    return [Vector.from_tensor(tensor).to_cocotb() for tensor in tensor_seq]


def cocotb_seq_to_tensor(cocotb_seq, mapping: TileMapping):
    tensor_seq = torch.stack(
        [
            Vector.from_cocotb(int_value, mapping.num, mapping.dtype).to_tensor()
            for int_value in cocotb_seq
        ]
    )
    tensor = einops.rearrange(
        tensor_seq,
        mapping.hw_to_sw_formula,
        **mapping.hw_temp_dim,
        **mapping.hw_spat_dim,
    )  # [N, M]
    return tensor
