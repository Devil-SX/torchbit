"""
Test case: pipeline correctness under backpressure

Uses RandomBackpressure strategy to insert random stall cycles on the
driver side, verifying that the pipeline still produces correct output
regardless of input timing.
"""
import sys
from pathlib import Path

TB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TB_ROOT))

import cocotb
import torch
from cocotb.triggers import ClockCycles, Event

from tb.dut_pipe.wrapper.dut_wrapper import PipeUvmWrapper
from tb.common.golden_model.pipe_golden import PipeGolden
from torchbit.tools.strategy import RandomBackpressure


@cocotb.test()
async def test_backpressure(dut):
    """Drive random data with 30% stall probability, verify output."""
    dut._log.info("[Test] Starting backpressure pipeline test")

    wrapper = PipeUvmWrapper(dut, delay=4, debug=False)
    await wrapper.init()

    golden = PipeGolden(delay=4)

    # Random test data
    torch.manual_seed(42)
    sequence = torch.randint(0, 1000, (30,), dtype=torch.int32).tolist()
    dut._log.info(f"[Test] Sequence ({len(sequence)} values): {sequence[:5]}...")

    # Load expected
    expected = golden.predict(sequence)
    wrapper.load_expected(expected)

    # Start monitor
    stop_event = Event()
    monitor_task = cocotb.start_soon(wrapper.monitor_output(stop_event))

    # Drive with random backpressure (30% stall, deterministic seed)
    strategy = RandomBackpressure(stall_prob=0.3, seed=123)
    await wrapper.drive_sequence(sequence, strategy=strategy)

    # Extra flush time to account for stall gaps
    await ClockCycles(dut.clk, wrapper.delay + 10)

    stop_event.set()
    await monitor_task

    # Verify
    report = wrapper.scoreboard.report()
    dut._log.info(f"[Test] {report}")
    assert wrapper.scoreboard.passed, f"Scoreboard FAILED:\n{report}"

    dut._log.info("[Test] Backpressure pipeline test PASSED!")
