"""
Test case: basic pipeline correctness

Drives a simple ascending sequence through the pipe, waits for the
pipeline to flush, then verifies via the UVM scoreboard that every
output value matches the golden model prediction.
"""
import sys
from pathlib import Path

TB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TB_ROOT))

import cocotb
from cocotb.triggers import ClockCycles, Event

from tb.dut_pipe.wrapper.dut_wrapper import PipeUvmWrapper
from tb.common.golden_model.pipe_golden import PipeGolden


@cocotb.test()
async def test_basic(dut):
    """Drive 1..20 through pipe and verify with scoreboard."""
    dut._log.info("[Test] Starting basic pipeline test")

    wrapper = PipeUvmWrapper(dut, delay=4, debug=True)
    await wrapper.init()

    golden = PipeGolden(delay=4)

    # Test sequence
    sequence = list(range(1, 21))  # [1, 2, ..., 20]

    # Load expected results into scoreboard
    expected = golden.predict(sequence)
    wrapper.load_expected(expected)

    # Start monitor
    stop_event = Event()
    monitor_task = cocotb.start_soon(wrapper.monitor_output(stop_event))

    # Drive stimulus
    await wrapper.drive_sequence(sequence)

    # Wait for pipeline flush
    await ClockCycles(dut.clk, wrapper.delay + 5)

    # Stop monitor
    stop_event.set()
    await monitor_task

    # Check scoreboard
    report = wrapper.scoreboard.report()
    dut._log.info(f"[Test] {report}")
    assert wrapper.scoreboard.passed, f"Scoreboard FAILED:\n{report}"

    dut._log.info("[Test] Basic pipeline test PASSED!")
