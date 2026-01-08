"""
Test case for overlapping regions

Tests the MemoryMover DUT when source and destination regions overlap.
"""
import sys
from pathlib import Path

# Add testbench root to path
TB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TB_ROOT))

import cocotb
import torch
from cocotb.triggers import ClockCycles

from tb.dut_buffer.wrapper.dut_wrapper import MemoryMoverWrapper
from tb.common.golden_model.memory_mover_golden import MemoryMoverGolden


@cocotb.test()
async def test_copy_overlap(dut):
    """Test copy with overlapping regions (ascending)."""
    dut._log.info("[Test] Starting overlap copy test")

    # Initialize wrapper
    wrapper = MemoryMoverWrapper(dut)
    await wrapper.init()

    # Initialize golden model
    golden = MemoryMoverGolden()

    # Test parameters - overlapping regions
    # Copy from address 10 to address 5 (ascending copy: reads from lower addresses first)
    src_base = 10
    dst_base = 5
    num_words = 16

    # Create source data (sequential values for verification)
    src_data = torch.arange(100, 100 + num_words, dtype=torch.int32)
    dut._log.info(f"[Test] Source data: {src_data.tolist()}")

    # Load data
    wrapper.load_source_data(src_data, base_addr=src_base)
    golden.load_data(src_data, base_addr=src_base)

    await ClockCycles(dut.clk, 10)

    # Run copy
    await wrapper.run_copy(src_base, dst_base, num_words)

    # Read and verify
    dut_result = wrapper.read_destination_data(num_words, base_addr=dst_base)
    golden_result = golden.copy(src_base, dst_base, num_words)

    assert torch.equal(dut_result, golden_result), \
        f"Overlap copy failed! Expected {golden_result.tolist()}, got {dut_result.tolist()}"

    dut._log.info("[Test] Overlap copy test PASSED!")
