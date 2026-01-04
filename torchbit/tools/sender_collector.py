"""
Data driver and collector utilities for Cocotb testbenches.

Provides Sender, PoolCollector, and FIFOCollector classes for driving
stimulus into and collecting responses from HDL interfaces.
"""
import cocotb
from cocotb.triggers import RisingEdge, Event, Timer
from cocotb.utils import get_sim_time
from cocotb.handle import Immediate
from .port import InputPort, OutputPort
from typing import List, Optional, Any


class Sender:
    """Data driver for sending stimuli to HDL interfaces.

    A Cocotb-aware driver that sequences data values and drives them
    onto HDL signals. Supports flow control via full/ready signals.

    The Sender:
    - Maintains a queue of values to send
    - Drives data and valid signals on each clock cycle
    - Optionally waits for backpressure (full/ready) signals
    - Records timestamps for each sent value

    Attributes:
        debug (bool): Enable debug logging.
        queue (list): Data values pending to be sent.
        timestamps (list): Simulation timestamps when data was sent.
        data_port (OutputPort): Connected data signal.
        valid_port (OutputPort): Connected valid signal.
        full_port (InputPort or None): Connected full/backpressure signal.

    Example:
        >>> from torchbit.tools import Sender
        >>> sender = Sender(debug=True)
        >>> sender.connect(dut=dut, clk=dut.clk, data=dut.data, valid=dut.valid)
        >>> sender.load([0x1, 0x2, 0x3, 0x4])
        >>> cocotb.start_soon(sender.run())

    With flow control:
        >>> sender = Sender()
        >>> sender.connect(dut=dut, clk=dut.clk, data=dut.data,
        ...                valid=dut.valid, full=dut.fifo_full)
        >>> sender.load([0x1, 0x2, 0x3, 0x4])
        >>> cocotb.start_soon(sender.run())
        >>>
        >>> # Later, stop the sender
        >>> stop_event = Event()
        >>> stop_event.set()
    """

    def __init__(self, debug: bool = False):
        """Initialize a Sender.

        Args:
            debug: If True, enable debug logging for each sent value.
        """
        self.debug = debug
        self.queue: List[int] = []
        self.timestamps: List[tuple] = []

    def connect(self, dut, clk, data, valid, full=None) -> None:
        """Connect the Sender to HDL signals.

        Args:
            dut: The DUT object.
            clk: Clock signal.
            data: Data output signal.
            valid: Valid output signal.
            full: (optional) Backpressure signal. Interpretation depends on
                  protocol: for FIFO, 1 means stop; for ready/valid, 0 means stop.
        """
        self.dut = dut
        self.clk = clk
        self.data_port = OutputPort(data)
        self.valid_port = OutputPort(valid)
        self.full_port = InputPort(full) if full is not None else None

    def load(self, sequence: List[int]) -> None:
        """Load a sequence of values to send.

        Args:
            sequence: List of integer values to send in order.
        """
        self.queue = list(sequence)
        self.timestamps = []

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each sent value.

        Returns:
            List of (time_value, time_unit) tuples for each sent value.
        """
        return self.timestamps

    async def run(self, stop_event: Event = None) -> None:
        """Start sending data.

        Runs as a coroutine, sending values from the queue on each
        clock cycle until all values are sent or stop_event is set.

        Args:
            stop_event: Optional Event to signal early termination.
        """
        idx = 0
        while idx < len(self.queue):
            if stop_event is not None and isinstance(stop_event, Event):
                if stop_event.is_set():
                    break

            # Drive signals
            self.data_port.set(self.queue[idx])
            self.valid_port.set(1)  # Always Ready logic

            # Wait for clock edge
            await RisingEdge(self.clk)

            # Check flow control to decide if we advance
            can_send = True
            if self.full_port is not None:
                r_val = self.full_port.get()
                # If full signal: 1 means Stop
                # If normal READY: 0 means Stop
                can_send = (r_val == 0)

            if can_send:
                t = get_sim_time()
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Sender] Sent {self.queue[idx]} at {t}")
                idx += 1

        # Done, deassert valid
        self.valid_port.set(0)


class PoolCollector:
    """Collects data from HDL interfaces without flow control.

    A simple collector that captures data when valid is asserted.
    Assumes always-ready reception of data (no backpressure).

    The PoolCollector:
    - Monitors data and valid signals
    - Captures data on each valid cycle
    - Records timestamps for each captured value
    - Runs until stop_event is set

    Attributes:
        debug (bool): Enable debug logging.
        data (list): Collected data values.
        timestamps (list): Simulation timestamps when data was captured.

    Example:
        >>> from torchbit.tools import PoolCollector
        >>> collector = PoolCollector(debug=True)
        >>> collector.connect(dut=dut, clk=dut.clk, data=dut.data, valid=dut.valid)
        >>> stop_event = Event()
        >>> cocotb.start_soon(collector.run(stop_event))
        >>>
        >>> # Run simulation...
        >>>
        >>> # Stop collection
        >>> stop_event.set()
        >>> results = collector.dump()
        >>> times = collector.dump_time()
    """

    def __init__(self, debug: bool = False):
        """Initialize a PoolCollector.

        Args:
            debug: If True, enable debug logging for each captured value.
        """
        self.debug = debug
        self.data: List[int] = []
        self.timestamps: List[tuple] = []

    def connect(self, dut, clk, data, valid) -> None:
        """Connect the PoolCollector to HDL signals.

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
        # Collector now implicitly assumes it's always ready to receive,
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
                    self.dut._log.info(f"[Collector] Collected {val} at {t}")

    def dump(self) -> List[int]:
        """Get all collected data values.

        Returns:
            List of captured data values in order.
        """
        return self.data

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each collected value.

        Returns:
            List of (time_value, time_unit) tuples for each captured value.
        """
        return self.timestamps


class FIFOCollector:
    """Collects data with FIFO-style ready/empty flow control.

    Drives the ready signal based on FIFO empty status and captures
    data when both valid and ready are asserted.

    The FIFOCollector:
    - Monitors empty signal to determine if FIFO has data
    - Asserts ready when FIFO is not empty
    - Captures data on valid+ready handshake
    - Records timestamps for each captured value

    Attributes:
        debug (bool): Enable debug logging.
        data (list): Collected data values.
        timestamps (list): Simulation timestamps.

    Example:
        >>> from torchbit.tools import FIFOCollector
        >>> collector = FIFOCollector(debug=True)
        >>> collector.connect(dut=dut, clk=dut.clk,
        ...                   data=dut.data, empty=dut.fifo_empty,
        ...                   ready=dut.fifo_ready)
        >>> stop_event = Event()
        >>> cocotb.start_soon(collector.run(stop_event))
    """

    def __init__(self, debug: bool = False):
        """Initialize a FIFOCollector.

        Args:
            debug: If True, enable debug logging.
        """
        self.debug = debug
        print(f"[FIFOCollector] Initialized {debug=}")
        self.data: List[int] = []
        self.timestamps: List[tuple] = []

    def connect(self, dut, clk, data, empty, ready, valid=None) -> None:
        """Connect the FIFOCollector to HDL signals.

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
        # Collector now implicitly assumes it's always ready to receive,
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
                self.dut._log.info(f"[Collector] Valid {v_val} at {t}")

            is_valid = (v_val == 1)  # Hardcoded to check for active-high valid

            if is_valid:
                val = self.data_port.get()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[Collector] Collected {val} at {t}")

            last_ready = cur_ready

        # Cleanup: No ready_port to set to 0.

    def dump(self) -> List[int]:
        """Get all collected data values.

        Returns:
            List of captured data values in order.
        """
        return self.data

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each collected value.

        Returns:
            List of (time_value, time_unit) tuples for each captured value.
        """
        return self.timestamps
