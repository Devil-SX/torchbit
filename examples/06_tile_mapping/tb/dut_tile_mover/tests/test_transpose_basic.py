"""
Test case for basic b,w,c -> w,b,c transpose operation

Tests the basic functionality of the TileMover DUT:
1. Load source tensor (b, w, c) using TwoPortBuffer.init_from_tensor (backdoor write)
2. Run transpose operation (b, w, c) -> (w, b, c)
3. Read destination tensor (w, b, c) using TwoPortBuffer.dump_to_tensor (backdoor read)
4. Compare with golden model output (einops rearrange)

Demonstrates spatial mapping:
- DATA_WIDTH = 64 bits
- dtype = int16 (16 bits)
- cs = 4 (spatial dimension - 4 elements transferred per cycle)
"""
import sys
from pathlib import Path

# Add testbench root to path
TB_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TB_ROOT))

import cocotb
import torch
import einops
from cocotb.triggers import ClockCycles

from tb.dut_tile_mover.wrapper.tile_mover_wrapper import TileMoverWrapper
from tb.common.golden_model.tile_mover_golden import TileMoverGolden


@cocotb.test()
async def test_transpose_basic(dut):
    """Test basic transpose operation (b,w,c) -> (w,b,c)."""
    dut._log.info("[Test] Starting basic transpose test (2,4,8) -> (4,2,8)")

    # Initialize wrapper
    wrapper = TileMoverWrapper(dut, debug=False)
    await wrapper.init()

    # Initialize golden model
    golden = TileMoverGolden()

    # Test parameters
    # b=2 (batch), w=4 (width), c=8 (channel)
    # cs=4 (spatial), so ct = c/cs = 8/4 = 2 (temporal)
    b, w, c = 2, 4, 8
    src_base = 0x00
    dst_base = 0x40

    # Create source tensor (b, w, c) with sequential values
    src_tensor = torch.arange(b * w * c, dtype=torch.int16).reshape(b, w, c)
    dut._log.info(f"[Test] Source tensor ({b}x{w}x{c}):\n{src_tensor}")

    # Load data into DUT via TwoPortBuffer.init_from_tensor
    wrapper.load_source_tensor(src_tensor, b, w, c, base_addr=src_base)

    # Wait for writes to complete
    await ClockCycles(dut.clk, 10)

    # Run DUT transpose operation
    dut._log.info(f"[Test] Running transpose: {src_base:#x} -> {dst_base:#x}, ({b},{w},{c}) -> ({w},{b},{c})")
    await wrapper.run_transpose(src_base, dst_base, b, w, c)

    # Read destination tensor from DUT via TwoPortBuffer.dump_to_tensor
    dut_result = wrapper.read_destination_tensor(b, w, c, base_addr=dst_base)
    dut._log.info(f"[Test] DUT result ({w}x{b}x{c}):\n{dut_result}")

    # Get golden model result using einops rearrange
    golden_tensor = golden.transpose(src_tensor, b, w, c)
    dut._log.info(f"[Test] Golden result ({w}x{b}x{c}):\n{golden_tensor}")

    # Verify against golden model
    assert torch.equal(dut_result, golden_tensor), \
        f"Data mismatch!\nExpected:\n{golden_tensor}\nGot:\n{dut_result}"

    # Also verify against PyTorch einops
    expected = einops.rearrange(src_tensor, "b w c -> w b c")
    assert torch.equal(dut_result, expected), \
        f"Transpose mismatch with einops!\nExpected:\n{expected}\nGot:\n{dut_result}"

    dut._log.info("[Test] Basic transpose test PASSED!")
