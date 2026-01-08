"""
Test case for TwoPortBuffer roundtrip

Tests TwoPortBuffer write and read operations directly.
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


@cocotb.test()
async def test_buffer_roundtrip(dut):
    """Test TwoPortBuffer roundtrip: write -> read -> verify."""
    dut._log.info("[Test] Starting buffer roundtrip test")

    # Initialize wrapper
    wrapper = MemoryMoverWrapper(dut)
    await wrapper.init()

    # Test data
    test_data = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32)
    dut._log.info(f"[Test] Test data: {test_data.tolist()}")

    # Write data via TwoPortBuffer
    wrapper.load_source_data(test_data, base_addr=0)

    # Wait for writes
    await ClockCycles(dut.clk, 10)

    # Read back via TwoPortBuffer
    result = wrapper.read_destination_data(len(test_data), base_addr=0)
    dut._log.info(f"[Test] Read back: {result.tolist()}")

    # Verify
    assert torch.equal(test_data, result), \
        f"Roundtrip failed! Expected {test_data.tolist()}, got {result.tolist()}"

    dut._log.info("[Test] Buffer roundtrip test PASSED!")
