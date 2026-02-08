"""
Data monitor utilities for Cocotb testbenches.

Provides PoolMonitor and FIFOMonitor classes for collecting
responses from HDL interfaces.
"""
import cocotb
from cocotb.triggers import RisingEdge, Event
from cocotb.utils import get_sim_time
from cocotb.handle import Immediate
from .port import InputPort, OutputPort
from ..core.logic_sequence import LogicSequence
from typing import List


class PoolMonitor:
    """Collects data from HDL interfaces without flow control.

    A simple monitor that captures data when valid is asserted.
    Assumes always-ready reception of data (no backpressure).

    The PoolMonitor:
    - Monitors data and valid signals
    - Captures data on each valid cycle
    - Records timestamps for each captured value
    - Runs until stop_event is set

    Attributes:
        debug (bool): Enable debug logging.
        data (LogicSequence): Collected data values.
        timestamps (list): Simulation timestamps when data was captured.

    Example:
        >>> from torchbit.tools import PoolMonitor
        >>> monitor = PoolMonitor(debug=True)
        >>> monitor.connect(dut=dut, clk=dut.clk, data=dut.data, valid=dut.valid)
        >>> stop_event = Event()
        >>> cocotb.start_soon(monitor.run(stop_event))
        >>>
        >>> # Run simulation...
        >>>
        >>> # Stop collection
        >>> stop_event.set()
        >>> results = monitor.dump()
        >>> times = monitor.dump_time()
    """

    def __init__(self, debug: bool = False):
        """Initialize a PoolMonitor.

        Args:
            debug: If True, enable debug logging for each captured value.
        """
        self.debug = debug
        self.data: LogicSequence = LogicSequence()
        self.timestamps: List[tuple] = []

    def connect(self, dut, clk, data, valid) -> None:
        """Connect the PoolMonitor to HDL signals.

        Args:
            dut: The DUT object.
            clk: Clock signal.
            data: Data input signal.
            valid: Valid input signal.
        """
        self.dut = dut
        self.clk = clk
        self.data_port = InputPort(data)
        self.valid_port = InputPort(valid)

    async def run(self, stop_event: Event) -> None:
        """Start collecting data.

        Runs as a coroutine, capturing data on each valid cycle
        until stop_event is set.

        Args:
            stop_event: Event that signals when to stop collection.
        """
        # Monitor now implicitly assumes it's always ready to receive,
        # as 'ready' output was removed.

        while not stop_event.is_set():
            await RisingEdge(self.clk)

            # Check valid
            # Receive Logic
            v_val = self.valid_port.get()
            is_valid = (v_val == 1)  # Hardcoded to check for active-high valid

            if is_valid:
                val = self.data_port.get()
                t = get_sim_time()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Monitor] Collected {val} at {t}")

    def dump(self) -> LogicSequence:
        """Get all collected data values.

        Returns:
            LogicSequence of captured data values in order.
        """
        return self.data

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each collected value.

        Returns:
            List of (time_value, time_unit) tuples for each captured value.
        """
        return self.timestamps


class FIFOMonitor:
    """Collects data with FIFO-style ready/empty flow control.

    Drives the ready signal based on FIFO empty status and captures
    data when both valid and ready are asserted.

    The FIFOMonitor:
    - Monitors empty signal to determine if FIFO has data
    - Asserts ready when FIFO is not empty
    - Captures data on valid+ready handshake
    - Records timestamps for each captured value

    Attributes:
        debug (bool): Enable debug logging.
        data (list): Collected data values.
        timestamps (list): Simulation timestamps.

    Example:
        >>> from torchbit.tools import FIFOMonitor
        >>> monitor = FIFOMonitor(debug=True)
        >>> monitor.connect(dut=dut, clk=dut.clk,
        ...                   data=dut.data, empty=dut.fifo_empty,
        ...                   ready=dut.fifo_ready)
        >>> stop_event = Event()
        >>> cocotb.start_soon(monitor.run(stop_event))
    """

    def __init__(self, debug: bool = False):
        """Initialize a FIFOMonitor.

        Args:
            debug: If True, enable debug logging.
        """
        self.debug = debug
        print(f"[FIFOMonitor] Initialized {debug=}")
        self.data: LogicSequence = LogicSequence()
        self.timestamps: List[tuple] = []

    def connect(self, dut, clk, data, empty, ready, valid=None) -> None:
        """Connect the FIFOMonitor to HDL signals.

        Args:
            dut: The DUT object.
            clk: Clock signal.
            data: Data input signal.
            empty: FIFO empty signal (1=empty, 0=has data).
            ready: FIFO ready output signal.
            valid: (optional) Valid input signal. If None, uses ready
                  latched value as valid (handshake after ready).
        """
        self.dut = dut
        self.clk = clk
        self.data_port = InputPort(data)

        self.empty_port = InputPort(empty)
        self.ready_port = OutputPort(ready)
        self.valid_port = InputPort(valid) if valid is not None else None

    async def run(self, stop_event: Event) -> None:
        """Start collecting data.

        Runs as a coroutine, managing ready/deassertion and capturing
        data on valid/ready handshake until stop_event is set.

        Args:
            stop_event: Event that signals when to stop collection.
        """
        # Monitor now implicitly assumes it's always ready to receive,
        # as 'ready' output was removed.
        self.ready_port.set(Immediate(0))

        last_ready = 0
        cur_ready = 0
        while not stop_event.is_set():
            await RisingEdge(self.clk)
            t = get_sim_time()

            empty = self.empty_port.get()
            if not empty:
                self.ready_port.set(Immediate(1))
                cur_ready = 1
            else:
                self.ready_port.set(Immediate(0))
                cur_ready = 0

            # Check valid
            # Receive Logic
            if self.valid_port is not None:
                v_val = self.valid_port.get()
            else:
                # Valid set after next ready
                v_val = last_ready
            if self.debug:
                self.dut._log.info(f"[Monitor] Valid {v_val} at {t}")

            is_valid = (v_val == 1)  # Hardcoded to check for active-high valid

            if is_valid:
                val = self.data_port.get()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Monitor] Collected {val} at {t}")

            last_ready = cur_ready

        # Cleanup: No ready_port to set to 0.

    def dump(self) -> LogicSequence:
        """Get all collected data values.

        Returns:
            LogicSequence of captured data values in order.
        """
        return self.data

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each collected value.

        Returns:
            List of (time_value, time_unit) tuples for each captured value.
        """
        return self.timestamps
