from pathlib import Path
from os import PathLike
import torch
import numpy as np
import einops
from ..core.hw_vector import HwVector
from .buffer import Buffer
from ..debug.judge import compare

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


class TestVector:
    def __init__(
        self,
        source: PathLike | np.ndarray | torch.Tensor,
        source_format: str,
        target_format: str,
        temp_dim: dict,
        spat_dim: dict,
        base=0,
        strides: dict = None,
    ):
        match source:
            case PathLike():
                self.source = torch.from_numpy(np.load(source))
            case np.ndarray():
                self.source = torch.from_numpy(source)
            case torch.Tensor():
                self.source = source
            case _:
                raise ValueError("source must be a path or numpy array or torch tensor")
        self.trans_formula = f"{source_format} -> {target_format}"
        self.temp_dim = temp_dim
        self.spat_dim = spat_dim

        if strides is not None:
            self.address_mapping = AddressMapping(
                base, tuple(strides.values()), tuple(temp_dim.values())
            )

    def to_cocotb_seq(self):
        tensor_seq = einops.rearrange(
            self.source, self.trans_formula, **self.temp_dim, **self.spat_dim
        )  # [N, M]
        return [HwVector.from_tensor(tensor).to_cocotb() for tensor in tensor_seq]

    def init_buf(self, buf: Buffer):
        addr_list = self.address_mapping.get_addr_list()
        tensor_seq = einops.rearrange(
            self.source, self.trans_formula, **self.temp_dim, **self.spat_dim
        )  # [N, M]

        for tensor, addr in zip(tensor_seq, addr_list):
            buf.write(addr, HwVector.from_tensor(tensor).to_cocotb())


class GroundTrueResult:
    def __init__(
        self,
        target: PathLike | np.ndarray | torch.Tensor,
        source_format: str,
        target_format: str,
        dtype: torch.dtype,
        num: int,
        temp_dim: dict,
        spat_dim: dict,
        base=0,
        strides: dict = None,
    ):
        match target:
            case PathLike():
                self.target = torch.from_numpy(np.load(target))
            case np.ndarray():
                self.target = torch.from_numpy(target)
            case torch.Tensor():
                self.target = target
            case _:
                raise ValueError("source must be a path or numpy array or torch tensor")
        self.trans_formula = f"{source_format} -> {target_format}"
        self.temp_dim = temp_dim
        self.spat_dim = spat_dim
        self.dtype = dtype
        self.num = num

        if strides is not None:
            self.address_mapping = AddressMapping(
                base, tuple(strides.values()), tuple(temp_dim.values())
            )

    def to_tensor(self):
        return self.test_tensor

    def from_cocotb_seq(self, cocotb_seq):
        tensor_seq = torch.stack(
            [
                HwVector.from_cocotb(int_value, self.num, self.dtype).to_tensor()
                for int_value in cocotb_seq
            ]
        )
        tensor = einops.rearrange(
            tensor_seq, self.trans_formula, **self.temp_dim, **self.spat_dim
        )  # [N, M]
        self.test_tensor = tensor

    def from_buf_dump_seq(self, buf):
        addr_list = self.address_mapping.get_addr_list()
        cocotb_seq = [buf.read(addr) for addr in addr_list]
        tensor_seq = [
            HwVector.from_cocotb(int_value, self.num, self.dtype).to_tensor()
            for int_value in cocotb_seq
        ]
        tensor = einops.rearrange(
            tensor_seq, self.trans_formula, **self.temp_dim, **self.spat_dim
        )  # [N, M]
        self.test_tensor = tensor

    def judge(self):
        compare(self.test_tensor, self.target)
