"""
Test case for configurable transpose dimensions

Tests that the TileMover DUT correctly handles different dimensions
configured via the instruction port at runtime. This demonstrates
the flexibility of the instruction-based configuration approach.
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
async def test_transpose_configurable(dut):
    """Test transpose with various dimensions via instruction port."""
    dut._log.info("[Test] Starting configurable dimensions test")

    # Initialize wrapper
    wrapper = TileMoverWrapper(dut, debug=False)
    await wrapper.init()

    # Initialize golden model
    golden = TileMoverGolden()

    # Test various dimensions - all configured via instruction port
    # c must be multiple of cs=4 (spatial dimension)
    test_cases = [
        # (b, w, c, src_base, dst_base, description)
        (1, 2, 4, 0x00, 0x10, "1x2x4 (minimal)"),
        (2, 2, 4, 0x00, 0x10, "2x2x4 square"),
        (2, 4, 8, 0x00, 0x40, "2x4x8 default"),
        (2, 4, 12, 0x00, 0x60, "2x4x12 rectangular"),
        (4, 2, 8, 0x00, 0x40, "4x2x8 swapped"),
        (1, 8, 16, 0x00, 0x80, "1x8x16 wide"),
        (8, 1, 4, 0x00, 0x20, "8x1x4 tall"),
    ]

    for b, w, c, src_base, dst_base, desc in test_cases:
        dut._log.info(f"[Test] Testing {desc}: ({b},{w},{c}), src={src_base:#x}, dst={dst_base:#x}")

        # Create source tensor with sequential values
        src_tensor = torch.arange(b * w * c, dtype=torch.int16).reshape(b, w, c)
        dut._log.info(f"[Test] Source tensor shape: {src_tensor.shape}")

        # Load data via init_from_tensor
        wrapper.load_source_tensor(src_tensor, b, w, c, base_addr=src_base)

        await ClockCycles(dut.clk, 5)

        # Run transpose with dimensions configured via instruction port
        await wrapper.run_transpose(src_base, dst_base, b, w, c)

        # Read result via dump_to_tensor
        dut_result = wrapper.read_destination_tensor(b, w, c, base_addr=dst_base)

        # Verify against golden model (einops rearrange)
        expected = golden.transpose(src_tensor, b, w, c)

        assert dut_result.shape == expected.shape, \
            f"Shape mismatch! Expected {expected.shape}, got {dut_result.shape}"
        assert torch.equal(dut_result, expected), \
            f"{desc} transpose failed!\nExpected:\n{expected}\nGot:\n{dut_result}"

        dut._log.info(f"[Test] {desc} PASSED")

    dut._log.info("[Test] All configurable dimensions tests PASSED!")


@cocotb.test()
async def test_transpose_runtime_reconfig(dut):
    """Test that dimensions can be changed at runtime between operations."""
    dut._log.info("[Test] Starting runtime reconfiguration test")

    # Initialize wrapper
    wrapper = TileMoverWrapper(dut, debug=False)
    await wrapper.init()

    # Initialize golden model
    golden = TileMoverGolden()

    # Test sequence: different configurations in consecutive operations
    operations = [
        # (b, w, c, src_base, dst_base)
        (2, 2, 4, 0x00, 0x10),
        (2, 4, 8, 0x00, 0x40),
        (1, 2, 4, 0x00, 0x10),
        (4, 2, 8, 0x00, 0x40),
    ]

    for i, (b, w, c, src_base, dst_base) in enumerate(operations):
        dut._log.info(f"[Test] Operation {i+1}: ({b},{w},{c})")

        # Create source tensor
        src_tensor = torch.arange(b * w * c, dtype=torch.int16).reshape(b, w, c) + (i * 1000)

        # Load and run
        wrapper.load_source_tensor(src_tensor, b, w, c, base_addr=src_base)
        await ClockCycles(dut.clk, 2)
        await wrapper.run_transpose(src_base, dst_base, b, w, c)

        # Read result
        dut_result = wrapper.read_destination_tensor(b, w, c, base_addr=dst_base)

        # Verify against golden model
        expected = golden.transpose(src_tensor, b, w, c)

        assert torch.equal(dut_result, expected), \
            f"Operation {i+1} failed! Expected {expected.flatten()}, got {dut_result.flatten()}"

    dut._log.info("[Test] Runtime reconfiguration test PASSED!")
