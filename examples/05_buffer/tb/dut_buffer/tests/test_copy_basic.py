"""
Test case for basic copy operation

Tests the basic functionality of the MemoryMover DUT:
1. Load source data using TwoPortBuffer (backdoor write)
2. Run copy operation
3. Read destination data using TwoPortBuffer (backdoor read)
4. Compare with golden model output
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
async def test_copy_basic(dut):
    """Test basic copy operation with small data set."""
    dut._log.info("[Test] Starting basic copy test")

    # Initialize wrapper
    wrapper = MemoryMoverWrapper(dut)
    await wrapper.init()

    # Initialize golden model
    golden = MemoryMoverGolden()

    # Test parameters
    src_base = 0
    dst_base = 128
    num_words = 16

    # Create source data
    src_data = torch.arange(num_words, dtype=torch.int32)
    dut._log.info(f"[Test] Source data: {src_data.tolist()}")

    # Load data into both DUT (via TwoPortBuffer) and golden model
    wrapper.load_source_data(src_data, base_addr=src_base)
    golden.load_data(src_data, base_addr=src_base)

    # Verify source data was loaded correctly (read back from buffer)
    src_verify = torch.zeros(num_words, dtype=torch.int32)
    for i in range(num_words):
        src_verify[i] = wrapper.buffer.read(src_base + i)
    dut._log.info(f"[Test] Source verify: {src_verify.tolist()}")
    assert torch.equal(src_verify, src_data), "Source data not loaded correctly!"

    # Wait for writes to complete
    await ClockCycles(dut.clk, 10)

    # Run DUT copy operation
    dut._log.info(f"[Test] Running copy: 0x{src_base:02X} -> 0x{dst_base:02X}, {num_words} words")
    await wrapper.run_copy(src_base, dst_base, num_words)

    # Read destination data from DUT
    dut_result = wrapper.read_destination_data(num_words, base_addr=dst_base)
    dut._log.info(f"[Test] DUT result: {dut_result.tolist()}")

    # Get golden model result
    golden_result = golden.copy(src_base, dst_base, num_words)
    dut._log.info(f"[Test] Golden result: {golden_result.tolist()}")

    # Verify
    assert torch.equal(dut_result, golden_result), \
        f"Data mismatch! Expected {golden_result.tolist()}, got {dut_result.tolist()}"

    # Also verify against original source data
    assert torch.equal(dut_result, src_data), \
        f"Data mismatch with source! Expected {src_data.tolist()}, got {dut_result.tolist()}"

    dut._log.info("[Test] Basic copy test PASSED!")
