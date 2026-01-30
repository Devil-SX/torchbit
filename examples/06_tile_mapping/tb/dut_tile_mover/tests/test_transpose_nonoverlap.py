"""
Test case for non-overlapping source and destination regions

Tests that the TileMover DUT correctly handles cases where source
and destination regions do not overlap.
"""
import sys
from pathlib import Path

# Add testbench root to path
TB_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TB_ROOT))

import cocotb
import torch
from cocotb.triggers import ClockCycles

from tb.dut_tile_mover.wrapper.tile_mover_wrapper import TileMoverWrapper
from tb.common.golden_model.tile_mover_golden import TileMoverGolden


@cocotb.test()
async def test_transpose_nonoverlap(dut):
    """Test transpose with non-overlapping regions."""
    dut._log.info("[Test] Starting non-overlap transpose test")

    # Initialize wrapper
    wrapper = TileMoverWrapper(dut, debug=False)
    await wrapper.init()

    # Initialize golden model
    golden = TileMoverGolden()

    # Test parameters - ensure non-overlapping
    b, w, c = 2, 4, 8
    src_base = 0x00
    dst_base = 0x40  # Well after source (2*4*8/4=16 addresses)

    # Verify non-overlapping
    src_size = b * w * c // wrapper.CS  # Number of memory addresses
    dst_size = b * w * c // wrapper.CS
    src_end = src_base + src_size
    dst_end = dst_base + dst_size
    is_overlap = not (dst_end <= src_base or src_end <= dst_base)
    assert not is_overlap, "Source and destination regions should not overlap!"
    dut._log.info(f"[Test] Non-overlapping regions: src={src_base:#x}, dst={dst_base:#x}")

    # Create source tensor
    src_tensor = torch.arange(b * w * c, dtype=torch.int16).reshape(b, w, c)
    dut._log.info(f"[Test] Source tensor ({b}x{w}x{c}):\n{src_tensor}")

    # Load data via init_from_tensor
    wrapper.load_source_tensor(src_tensor, b, w, c, base_addr=src_base)

    await ClockCycles(dut.clk, 10)

    # Run transpose
    await wrapper.run_transpose(src_base, dst_base, b, w, c)

    # Read result via dump_to_tensor
    dut_result = wrapper.read_destination_tensor(b, w, c, base_addr=dst_base)

    # Verify against golden model (einops rearrange)
    expected = golden.transpose(src_tensor, b, w, c)
    assert torch.equal(dut_result, expected), \
        f"Non-overlap transpose failed!\nExpected:\n{expected}\nGot:\n{dut_result}"

    dut._log.info("[Test] Non-overlap transpose test PASSED!")
