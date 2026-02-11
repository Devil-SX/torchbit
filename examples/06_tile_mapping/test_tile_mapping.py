"""
Tile Mapping Example - Cocotb Tests

Demonstrates TileMapping and AddressMapping for tensor-to-memory layout
with spatial/temporal dimension mapping.

DUT: tile_mover.v performs (b, w, c) -> (w, b, c) transpose using
a TwoPortBuffer as shared memory.

Spatial mapping:
- DATA_WIDTH = 64 bits, dtype = int16 (16 bits)
- cs = 4 elements transferred per cycle (spatial dimension)
- ct = c / cs (temporal dimension, requires multiple cycles)
"""
import cocotb
import torch
import einops
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from torchbit.tools import TwoPortBuffer
from torchbit.tiling import TileMapping, AddressMapping
from torchbit.core.dtype import dtype_to_bits
from torchbit.core.vector import Vector

CS = 4            # spatial dimension: 64-bit bus / 16-bit elements
DTYPE = torch.int16


async def init_dut(dut):
    """Reset and start clock."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def create_buffer(dut, debug=False):
    """Create and connect a TwoPortBuffer to the DUT."""
    buf = TwoPortBuffer(width=64, depth=256, debug=debug)
    buf.connect(
        dut=dut, clk=dut.clk,
        wr_csb=dut.wr_csb, wr_din=dut.wr_din, wr_addr=dut.wr_addr,
        rd_csb=dut.rd_csb, rd_addr=dut.rd_addr,
        rd_dout=dut.rd_dout, rd_dout_vld=dut.rd_dout_vld,
    )
    return buf


def load_source_tensor(buf, tensor, b, w, c, base_addr=0):
    """Load a (b, w, c) tensor into the buffer using TileMapping."""
    ct = c // CS
    mapping = TileMapping(
        dtype=DTYPE,
        sw_einops="b w (ct cs)",
        hw_einops="(b w ct) cs",
        hw_temp_dim={"b": b, "w": w, "ct": ct},
        hw_spat_dim={"cs": CS},
    )
    addr_mapping = AddressMapping(
        base=base_addr,
        hw_temp_einops="b w ct",
        hw_temp_dim={"b": b, "w": w, "ct": ct},
        hw_temp_stride={"b": w * ct, "w": ct, "ct": 1},
    )
    buf.backdoor_load_tensor(tensor.to(DTYPE), mapping, addr_mapping)


def read_dest_tensor(buf, b, w, c, base_addr=0):
    """Read a (w, b, c) transposed tensor from the buffer."""
    ct = c // CS
    num_addrs = w * b * ct
    tensor_2d = torch.zeros(num_addrs, CS, dtype=DTYPE)
    for i in range(num_addrs):
        vec = Vector.from_logic(buf.read(base_addr + i), CS, DTYPE)
        tensor_2d[i] = vec.to_array()
    return einops.rearrange(
        tensor_2d, "(w b ct) cs -> w b (ct cs)", w=w, b=b, ct=ct, cs=CS
    )


async def run_transpose(dut, src_base, dst_base, b, w, c):
    """Configure and run the DUT transpose."""
    dut.src_base_addr.value = src_base
    dut.dst_base_addr.value = dst_base
    dut.b_dim.value = b
    dut.w_dim.value = w
    dut.c_dim.value = c
    dut.start.value = 0
    await ClockCycles(dut.clk, 2)

    dut.start.value = 1
    await ClockCycles(dut.clk, 1)
    dut.start.value = 0

    timeout = (b * w * c // CS) * 2 + 100
    for _ in range(timeout):
        await ClockCycles(dut.clk, 1)
        if dut.done.value == 1:
            return
    raise TimeoutError(f"Tile mover timed out after {timeout} cycles!")


@cocotb.test()
async def test_transpose_basic(dut):
    """Basic (2,4,8) -> (4,2,8) transpose with golden model comparison."""
    await init_dut(dut)
    buf = create_buffer(dut)
    await buf.init()
    cocotb.start_soon(buf.run())

    b, w, c = 2, 4, 8
    src_tensor = torch.arange(b * w * c, dtype=DTYPE).reshape(b, w, c)

    load_source_tensor(buf, src_tensor, b, w, c, base_addr=0x00)
    await ClockCycles(dut.clk, 10)
    await run_transpose(dut, 0x00, 0x40, b, w, c)

    result = read_dest_tensor(buf, b, w, c, base_addr=0x40)
    expected = einops.rearrange(src_tensor, "b w c -> w b c")

    assert result.shape == expected.shape, f"Shape: {result.shape} != {expected.shape}"
    assert torch.equal(result, expected), f"Mismatch!\nExpected:\n{expected}\nGot:\n{result}"
    dut._log.info("test_transpose_basic PASSED")


@cocotb.test()
async def test_transpose_configurable(dut):
    """Test multiple dimension configurations."""
    await init_dut(dut)
    buf = create_buffer(dut)
    await buf.init()
    cocotb.start_soon(buf.run())

    cases = [
        (1, 2, 4,  0x00, 0x10, "1x2x4"),
        (2, 2, 4,  0x00, 0x10, "2x2x4"),
        (2, 4, 8,  0x00, 0x40, "2x4x8"),
        (4, 2, 8,  0x00, 0x40, "4x2x8"),
        (1, 8, 16, 0x00, 0x80, "1x8x16"),
    ]

    for b, w, c, src, dst, label in cases:
        src_tensor = torch.arange(b * w * c, dtype=DTYPE).reshape(b, w, c)
        load_source_tensor(buf, src_tensor, b, w, c, base_addr=src)
        await ClockCycles(dut.clk, 5)
        await run_transpose(dut, src, dst, b, w, c)
        result = read_dest_tensor(buf, b, w, c, base_addr=dst)
        expected = einops.rearrange(src_tensor, "b w c -> w b c")
        assert torch.equal(result, expected), f"{label} failed"
        dut._log.info(f"  {label} PASSED")

    dut._log.info("test_transpose_configurable PASSED")
