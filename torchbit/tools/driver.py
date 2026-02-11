"""
Data driver utility for Cocotb testbenches.

Provides the Driver class for driving stimulus into HDL interfaces.
"""
from .port import InputPort, OutputPort
from ..core.logic_sequence import LogicSequence
from typing import List


class Driver:
    """Data driver for sending stimuli to HDL interfaces.

    A Cocotb-aware driver that sequences data values and drives them
    onto HDL signals. Supports flow control via full/ready signals.

    The Driver:
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
        >>> from torchbit.tools import Driver
        >>> driver = Driver(debug=True)
        >>> driver.connect(dut=dut, clk=dut.clk, data=dut.data, valid=dut.valid)
        >>> driver.load([0x1, 0x2, 0x3, 0x4])
        >>> cocotb.start_soon(driver.run())

    With flow control:
        >>> driver = Driver()
        >>> driver.connect(dut=dut, clk=dut.clk, data=dut.data,
        ...                valid=dut.valid, full=dut.fifo_full)
        >>> driver.load([0x1, 0x2, 0x3, 0x4])
        >>> cocotb.start_soon(driver.run())
        >>>
        >>> # Later, stop the driver
        >>> stop_event = Event()
        >>> stop_event.set()
    """

    def __init__(self, debug: bool = False):
        """Initialize a Driver.

        Args:
            debug: If True, enable debug logging for each sent value.
        """
        self.debug = debug
        self.queue: LogicSequence = LogicSequence()
        self.timestamps: List[tuple] = []

    def connect(self, dut, clk, data, valid, full=None) -> None:
        """Connect the Driver to HDL signals.

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

    def load(self, sequence: LogicSequence) -> None:
        """Load a sequence of values to send.

        Args:
            sequence: LogicSequence of integer values to send in order.
        """
        self.queue = LogicSequence(sequence)
        self.timestamps = []

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each sent value.

        Returns:
            List of (time_value, time_unit) tuples for each sent value.
        """
        return self.timestamps

    async def run(self, stop_event=None) -> None:
        """Start sending data.

        Runs as a coroutine, sending values from the queue on each
        clock cycle until all values are sent or stop_event is set.

        Args:
            stop_event: Optional cocotb Event to signal early termination.
        """
        from cocotb.triggers import RisingEdge, Event
        from cocotb.utils import get_sim_time
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
                    self.dut._log.info(f"[Driver] Sent {self.queue[idx]} at {t}")
                idx += 1

        # Done, deassert valid
        self.valid_port.set(0)
