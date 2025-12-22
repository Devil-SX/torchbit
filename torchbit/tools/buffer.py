import cocotb
from cocotb.triggers import RisingEdge, Timer
import numpy as np
from ..core.vector import Vector
from ..core.dtype import dtype_to_bits
from .port import InputPort, OutputPort
from .mapping import TileMapping
from ..utils.bit_ops import replicate_bits
import torch
import einops
import copy


class Buffer:
    def __init__(self, width, depth):
        # width unit: bit
        exp2 = 2 ** (3 + np.arange(20))
        assert depth in exp2.astype(int).tolist(), "depth must be power of 2"
        self.addr_width = np.log2(depth).astype(int)
        self.width = width
        self.depth = depth
        self.clear()

    def write(self, addr, int_value):
        self.content[addr] = int_value

    def read(self, addr):
        return self.content[addr]

    def clear(self):
        self.content = [0] * self.depth

    def init_from_matrix(self, addr_start, addr_end, matrix):
        # [addr_start, addr_end)
        assert len(matrix.size()) == 2
        for i in range(addr_start, addr_end):
            self.content[i] = Vector.from_tensor(matrix[i - addr_start]).to_cocotb() & (
                (1 << self.width) - 1
            )

    def dump_to_matrix(self, addr_start, addr_end, dtype):
        # [addr_start, addr_end)

        if addr_end == self.depth:
            addr_end = -1

        sel_content = copy.deepcopy(self.content[addr_start:addr_end])
        num_bit = dtype_to_bits(dtype)

        content_tensor = []
        for data in sel_content:
            content_tensor.append(
                Vector.from_int(
                    value_int=data, num=self.width // num_bit, dtype=dtype
                ).to_tensor()
            )
        return torch.stack(content_tensor, dim=0)

    def init_from_tensor(self, tensor: torch.Tensor, mapping: TileMapping):
        addr_list = mapping.address_mapping.get_addr_list()
        tensor_seq = einops.rearrange(
            tensor,
            mapping.sw_to_hw_formula,
            **mapping.hw_temp_dim,
            **mapping.hw_spat_dim,
        )  # [N, M]

        for tensor, addr in zip(tensor_seq, addr_list):
            self.write(addr, Vector.from_tensor(tensor).to_cocotb())

    def dump_to_tensor(self, mapping: TileMapping):
        addr_list = mapping.address_mapping.get_addr_list()
        cocotb_seq = [self.read(addr) for addr in addr_list]
        tensor_seq = [
            Vector.from_cocotb(int_value, mapping.num, mapping.dtype).to_tensor()
            for int_value in cocotb_seq
        ]
        tensor = einops.rearrange(
            tensor_seq,
            mapping.hw_to_sw_formula,
            **mapping.hw_temp_dim,
            **mapping.hw_spat_dim,
        )  # [N, M]
        return tensor


def is_trig(value: int, is_pos_trig: bool):
    is_pos = value != 0
    return is_pos if is_pos_trig else not is_pos


class TwoPortBuffer(Buffer):
    def __init__(
        self,
        width,
        depth,
        backpressure=False,
        wrmask=False,
        debug=False,
        postive_trigger=False,
    ):
        # width as byte as unit
        super().__init__(width, depth)
        self.bp = backpressure
        self.debug = debug
        self.wrmask = wrmask
        self.is_pos_trig = postive_trigger

    def connect(
        self,
        dut,
        clk,
        wr_csb,
        wr_din,
        wr_addr,
        rd_csb,
        rd_addr,
        rd_dout,
        rd_dout_vld,
        wr_ready=None,
        rd_ready=None,
        wr_mask=None,
    ):
        self.dut = dut
        self.clk = clk
        self.wr_csb = InputPort(wr_csb)
        self.wr_din = InputPort(wr_din)
        self.wr_addr = InputPort(wr_addr)
        self.rd_csb = InputPort(rd_csb)
        self.rd_addr = InputPort(rd_addr)
        self.rd_dout = OutputPort(rd_dout)
        self.rd_dout_vld = OutputPort(rd_dout_vld)
        if self.bp:
            self.wr_ready = OutputPort(wr_ready, set_immediately=True)
            self.rd_ready = OutputPort(rd_ready, set_immediately=True)
        if self.wrmask:
            self.wr_mask = InputPort(wr_mask)

    async def init(self):
        self.rd_dout_vld.set(0)
        if self.bp:
            self.wr_ready.set(0)
            self.rd_ready.set(0)

    async def _run_read(self):
        while True:
            await RisingEdge(self.clk)
            read_trig = is_trig(self.rd_csb.get(), self.is_pos_trig)

            if read_trig:
                addr = self.rd_addr.get()
                self.rd_dout.set(self.content[addr])
                if self.debug:
                    self.dut._log.info(f"[TWBUF] read {self.content[addr]} from {addr}")
                self.rd_dout_vld.set(1)
            else:
                self.rd_dout_vld.set(0)

            if self.bp:
                self.rd_ready.set(1 if read_trig else 0)

    async def _run_write(self):
        while True:
            await RisingEdge(self.clk)
            write_trig = is_trig(self.wr_csb.get(), self.is_pos_trig)

            if self.bp:
                self.wr_ready.set(1 if write_trig else 0)

            if write_trig:
                addr = self.wr_addr.get() & ((1 << self.addr_width) - 1)
                data = self.wr_din.get() & ((1 << self.width) - 1)
                if self.wrmask:
                    wr_mask = replicate_bits(8, self.wr_mask.get())
                    data = (data & wr_mask) | (self.content[addr] & ~wr_mask)
                if self.debug:
                    self.dut._log.info(
                        f"[TWBUF] write {data}/{self.wr_din.get()} to {addr}"
                    )

                await Timer(1, "step")
                self.content[addr] = data

    async def run(self):
        cocotb.start_soon(self._run_read())
        cocotb.start_soon(self._run_write())
