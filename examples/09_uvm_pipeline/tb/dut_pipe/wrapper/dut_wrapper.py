"""
UVM-style DUT Wrapper for the Pipe module

Demonstrates how to build a verification wrapper using torchbit UVM
components:
- TorchbitBFM:        signal abstraction (drive/sample through named ports)
- TorchbitScoreboard: expected-vs-actual comparison via VectorItem
- TransferStrategy:   configurable driver timing (greedy, backpressure, burst)

No TileMapping or AddressMapping is used — the pipe operates on a
streaming valid/data protocol, not memory-mapped tensors.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

import torch
from torchbit.core import Vector
from torchbit.uvm import TorchbitBFM, VectorItem, TorchbitScoreboard
from torchbit.tools.strategy import GreedyStrategy


class PipeUvmWrapper:
    """UVM-style wrapper for the Pipe DUT.

    Signal access goes through TorchbitBFM; result checking goes through
    TorchbitScoreboard.  The wrapper exposes async helpers for driving
    stimulus and monitoring output that can be composed freely in tests.
    """

    def __init__(self, dut, delay: int = 4, debug: bool = False):
        """Initialize wrapper.

        Args:
            dut:   cocotb DUT handle.
            delay: Pipeline depth (must match RTL parameter DELAY).
            debug: Enable per-cycle log messages.
        """
        self.dut = dut
        self.delay = delay
        self.debug = debug
        self.clock = Clock(dut.clk, 10, unit="ns")

        # UVM components
        self.bfm = TorchbitBFM(dut=dut, clk=dut.clk)
        self.scoreboard = TorchbitScoreboard(name="pipe_scoreboard")

    async def init(self):
        """Start clock, register BFM signals, and apply reset."""
        cocotb.start_soon(self.clock.start())
        self.connect()

        # Drive idle state before reset
        self.bfm.drive('din_vld', 0)
        self.bfm.drive('din', 0)

        # Reset sequence
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, 3)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 2)

    def connect(self):
        """Register DUT signals in the BFM.

        Input group  (driven by testbench):  din, din_vld
        Output group (sampled by testbench): dout, dout_vld
        """
        self.bfm.add_input('din', self.dut.din)
        self.bfm.add_input('din_vld', self.dut.din_vld)
        self.bfm.add_output('dout', self.dut.dout)
        self.bfm.add_output('dout_vld', self.dut.dout_vld)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    async def drive_sequence(self, sequence, strategy=None):
        """Drive a list of integer values through the BFM.

        Args:
            sequence: List[int] — values to send.
            strategy: Optional TransferStrategy controlling per-cycle
                      stall behaviour. Defaults to GreedyStrategy
                      (no stalls).
        """
        if strategy is None:
            strategy = GreedyStrategy()

        cycle = 0
        for value in sequence:
            # Stall cycles (strategy says "not now")
            while not strategy.should_transfer(cycle):
                self.bfm.drive('din_vld', 0)
                await RisingEdge(self.dut.clk)
                cycle += 1

            # Transfer cycle
            self.bfm.drive('din', value)
            self.bfm.drive('din_vld', 1)
            if self.debug:
                self.dut._log.info(f"[Driver] Sent {value} at cycle {cycle}")
            await RisingEdge(self.dut.clk)
            cycle += 1

        # Deassert after last value
        self.bfm.drive('din_vld', 0)
        self.bfm.drive('din', 0)

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------
    async def monitor_output(self, stop_event):
        """Sample DUT output via BFM and feed each value to the scoreboard.

        Runs until *stop_event* is set. Returns the collected value list
        so that callers can inspect raw data if needed.

        Args:
            stop_event: cocotb Event — set to stop monitoring.

        Returns:
            List[int] of captured output values.
        """
        collected = []
        while not stop_event.is_set():
            await RisingEdge(self.dut.clk)
            if int(self.bfm.sample('dout_vld')) == 1:
                value = int(self.bfm.sample('dout'))
                item = VectorItem(
                    f"actual_{len(collected)}",
                    Vector.from_array(torch.tensor([value], dtype=torch.int32)),
                )
                self.scoreboard.add_actual(item)
                collected.append(value)
                if self.debug:
                    self.dut._log.info(f"[Monitor] Captured {value}")
        return collected

    # ------------------------------------------------------------------
    # Scoreboard helpers
    # ------------------------------------------------------------------
    def load_expected(self, golden_output):
        """Push expected values (from golden model) into the scoreboard.

        Args:
            golden_output: List[int] predicted by the golden model.
        """
        for i, value in enumerate(golden_output):
            item = VectorItem(
                f"expected_{i}",
                Vector.from_array(torch.tensor([value], dtype=torch.int32)),
            )
            self.scoreboard.add_expected(item)
