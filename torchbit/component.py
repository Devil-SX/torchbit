import cocotb
from cocotb.triggers import RisingEdge
import numpy as np
from .hensor import Hensor
from .dtype import dtype_to_bits
import torch
import copy


class InputPort:
    def __init__(self, wrapper_signal):
        self.wrapper = wrapper_signal

    def get(self):
        if self.wrapper is None:
            return 0
        return self.wrapper.value.integer


class OutputPort:
    def __init__(self, wrapper_signal):
        self.wrapper = wrapper_signal

    def set(self, value):
        if self.wrapper is None:
            return
        self.wrapper.value = value

class Buffer:
    def __init__(self, width, depth):
        # width unit: bit
        exp2 = 2 ** (3 + np.arange(20))
        assert depth in exp2.astype(int).tolist(), "depth must be power of 2"
        self.addr_width = np.log2(depth).astype(int)
        self.width = width
        self.depth = depth
        self.content = [0] * depth

    def write(self, addr, int_value):
        self.content[addr] = int_value
    
    def read(self, addr):
        return self.content[addr]

    def init_from_tensor(self, addr_start, addr_end, tensor):
        # [addr_start, addr_end)
        assert len(tensor.size()) == 2
        for i in range(addr_start, addr_end):
            self.content[i] = Hensor.from_tensor(tensor[i - addr_start]).to_cocotb() & ((1 << self.width) - 1)

    def dump_to_tensor(self, addr_start, addr_end, dtype):
        # [addr_start, addr_end)

        if addr_end == self.depth :
            addr_end = -1

        sel_content = copy.deepcopy(self.content[addr_start:addr_end])
        num_bit = dtype_to_bits(dtype) 

        content_tensor = []
        for data in sel_content:
            content_tensor.append(Hensor.from_int(value_int=data, num=self.width // num_bit, dtype=dtype).to_tensor())
        return torch.stack(content_tensor, dim=0)

        

def replicate_bits(num: int, n: int) -> int:
    assert num > 0
    
    # Convert number to binary string, removing '0b' prefix
    binary = bin(n)[2:]
    
    # Replicate each bit num times
    replicated = ''.join(bit * num for bit in binary)
    
    # Convert back to integer
    return int(replicated, 2)


class TwoPortBuffer(Buffer):
    def __init__(self, width, depth, backpressure=False, wrmask=False, debug=False, postive_trigger=False):
        # width as byte as unit
        super().__init__(width, depth)
        self.bp = backpressure
        self.debug = debug
        self.wrmask = wrmask
        self.trigger_signal = 1 if postive_trigger else 0

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
            self.wr_ready = OutputPort(wr_ready)
            self.rd_ready = OutputPort(rd_ready)
        if self.wrmask:
            self.wr_mask = InputPort(wr_mask)

    async def init(self):
        self.rd_dout_vld.set(0)
        if self.bp:
            self.wr_ready.set(1)
            self.rd_ready.set(1)


    async def run(self):
        while True:
            await RisingEdge(self.clk)
            # to implement: ready logic

            if self.rd_csb.get() == self.trigger_signal:
                addr = self.rd_addr.get()
                self.rd_dout.set(self.content[addr])
                if self.debug:
                    self.dut._log.info(f"[TWBUF] read {self.content[addr]} from {addr}")
                self.rd_dout_vld.set(1)
            else:
                self.rd_dout_vld.set(0)

            if self.wr_csb.get() == self.trigger_signal:  # Active low chip select
                addr = self.wr_addr.get() & ((1 << self.addr_width) - 1)
                data = self.wr_din.get() & ((1 << self.width) - 1)
                if self.wrmask:
                    wr_mask = replicate_bits(8, self.wr_mask.get())
                    data = (data & wr_mask) | (self.content[addr] & ~wr_mask)
                if self.debug:
                    self.dut._log.info(f"[TWBUF] write {data}/{self.wr_din.get()} to {addr}")

                self.content[addr] = data

