"""
Test case for larger copy operation

Tests the MemoryMover DUT with a larger data set.
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
async def test_copy_large(dut):
    """Test copy operation with larger data set (64 words)."""
    dut._log.info("[Test] Starting large copy test")

    # Initialize wrapper
    wrapper = MemoryMoverWrapper(dut)
    await wrapper.init()

    # Initialize golden model
    golden = MemoryMoverGolden()

    # Test parameters
    src_base = 0
    dst_base = 128
    num_words = 64

    # Create source data (random pattern)
    src_data = torch.randint(0, 1000, (num_words,), dtype=torch.int32)
    dut._log.info(f"[Test] Source data (first 10): {src_data[:10].tolist()}...")

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
        f"Large copy failed! Mismatch in results."

    dut._log.info("[Test] Large copy test PASSED!")
