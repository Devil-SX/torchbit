"""
Minimal cocotb test for simple_counter DUT.

This test demonstrates basic Runner functionality - compile and run
without extensive verification.
"""
import cocotb
from cocotb.triggers import Timer, RisingEdge, ClockCycles
from cocotb.clock import Clock


@cocotb.test
async def test_counter_runs(dut):
    """Basic test that runs the counter for a few cycles."""
    # Create a clock
    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    dut.enable.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Enable and run
    dut.enable.value = 1
    dut._log.info("[Test] Counter enabled, starting to count...")

    # Let it run for 10 cycles
    for i in range(10):
        await RisingEdge(dut.clk)
        count_val = dut.count.value.integer
        dut._log.info(f"[Test] Cycle {i}: count = {count_val}")

    # Disable
    dut.enable.value = 0
    await ClockCycles(dut.clk, 2)

    dut._log.info("[Test] Test completed successfully!")


@cocotb.test
async def test_counter_reset(dut):
    """Test that reset clears the counter."""
    clock = Clock(dut.clk, 1, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.rst_n.value = 0
    dut.enable.value = 1
    await ClockCycles(dut.clk, 2)

    # Release reset
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    count_val = dut.count.value.integer
    dut._log.info(f"[Test] After reset, count = {count_val}")

    # Should start from 0
    assert count_val == 0, f"Counter should be 0 after reset, got {count_val}"

    dut._log.info("[Test] Reset test passed!")
