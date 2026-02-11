"""FIFO interface components with pluggable transfer strategies.

Provides FIFODriver and FIFOReceiver for driving and capturing data
through FIFO-style HDL interfaces.  Both accept a TransferStrategy to
control transfer timing (greedy, random backpressure, burst, throttled).

These replace the original Driver (with ambiguous polarity) and
FIFOMonitor (which was actually a receiver) with clearer semantics.
The original classes are preserved for backward compatibility.
"""
from .port import InputPort, OutputPort
from .strategy import TransferStrategy, GreedyStrategy
from ..core.logic_sequence import LogicSequence
from typing import List


class FIFODriver:
    """Push data into a DUT input FIFO (TB -> DUT).

    Drives ``data`` and ``valid`` signals; reads ``ready`` for
    back-pressure.  A :class:`TransferStrategy` controls when
    ``valid`` is asserted, enabling greedy, burst, throttled,
    or random-backpressure patterns.

    Attributes:
        strategy (TransferStrategy): Controls transfer timing.
        debug (bool): Enable debug logging.
        queue (LogicSequence): Data values pending to be sent.
        timestamps (list): Simulation timestamps when data was sent.

    Example:
        >>> from torchbit.tools import FIFODriver, RandomBackpressure
        >>> driver = FIFODriver(strategy=RandomBackpressure(0.3))
        >>> driver.connect(dut=dut, clk=dut.clk,
        ...                data=dut.din, valid=dut.din_valid,
        ...                ready=dut.din_ready)
        >>> driver.load([0x1, 0x2, 0x3])
        >>> cocotb.start_soon(driver.run())
    """

    def __init__(self, strategy: TransferStrategy = None, debug: bool = False):
        """Initialize a FIFODriver.

        Args:
            strategy: Transfer strategy (defaults to GreedyStrategy).
            debug: If True, enable debug logging for each sent value.
        """
        self.strategy = strategy or GreedyStrategy()
        self.debug = debug
        self.queue: LogicSequence = LogicSequence()
        self.timestamps: List[tuple] = []
        self.tag = None

    @classmethod
    def from_path(cls, path: str, dut, clk, strategy=None, debug=False,
                  active_high=True):
        """Create a FIFODriver by resolving signals from ComponentDB.

        Args:
            path: ComponentDB path (e.g., ``"top.encoder.input"``).
            dut: DUT object.
            clk: Clock signal.
            strategy: TransferStrategy (defaults to GreedyStrategy).
            debug: Enable debug logging.
            active_high: Signal polarity for ready.

        Returns:
            A connected FIFODriver instance.
        """
        from .component_db import ComponentDB
        signals = ComponentDB.get(path)
        driver = cls(strategy=strategy, debug=debug)
        driver.tag = path.split(".")[-1]
        driver.connect(
            dut=dut, clk=clk,
            data=signals["data"],
            valid=signals["valid"],
            ready=signals.get("ready"),
            active_high=active_high,
        )
        ComponentDB.register_component(driver)
        return driver

    def connect(self, dut, clk, data, valid, ready=None, active_high: bool = True) -> None:
        """Connect the FIFODriver to HDL signals.

        Args:
            dut: The DUT object.
            clk: Clock signal.
            data: Data output signal (driven by driver).
            valid: Valid output signal (driven by driver).
            ready: (optional) Back-pressure input signal from DUT.
            active_high: If True, ready=1 means DUT can accept data.
                If False, ready=1 means DUT is full / cannot accept.
        """
        self.dut = dut
        self.clk = clk
        self.data_port = OutputPort(data)
        self.valid_port = OutputPort(valid)
        self.ready_port = InputPort(ready) if ready is not None else None
        self.active_high = active_high

    def load(self, sequence) -> None:
        """Load a sequence of values to send.

        Args:
            sequence: LogicSequence (or list of ints) to send in order.
        """
        self.queue = LogicSequence(sequence)
        self.timestamps = []

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each sent value.

        Returns:
            List of simulation time values for each sent value.
        """
        return self.timestamps

    async def run(self, stop_event=None) -> None:
        """Start sending data.

        Drives values from the queue on each clock cycle where the
        strategy allows and the DUT is ready, until all values are
        sent or ``stop_event`` is set.

        Args:
            stop_event: Optional cocotb Event to signal early termination.
        """
        from cocotb.triggers import RisingEdge
        from cocotb.utils import get_sim_time

        self.strategy.reset()
        idx = 0
        cycle = 0
        while idx < len(self.queue):
            if stop_event is not None and stop_event.is_set():
                break

            want = self.strategy.should_transfer(cycle)
            if want:
                self.data_port.set(self.queue[idx])
                self.valid_port.set(1)
            else:
                self.valid_port.set(0)

            await RisingEdge(self.clk)
            cycle += 1

            if not want:
                continue

            # Check back-pressure
            can_send = True
            if self.ready_port is not None:
                r_val = self.ready_port.get()
                can_send = bool(r_val) if self.active_high else not bool(r_val)

            if can_send:
                t = get_sim_time()
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[FIFODriver] Sent {self.queue[idx]} at {t}")
                idx += 1

        self.valid_port.set(0)


class FIFOReceiver:
    """Capture data from a DUT output FIFO (DUT -> TB).

    Reads ``data`` and ``valid`` signals; drives ``ready`` to accept
    data.  A :class:`TransferStrategy` controls when ``ready`` is
    asserted, enabling back-pressure injection from the consumer side.

    Attributes:
        strategy (TransferStrategy): Controls when ready is asserted.
        debug (bool): Enable debug logging.
        data (LogicSequence): Collected data values.
        timestamps (list): Simulation timestamps when data was captured.

    Example:
        >>> from torchbit.tools import FIFOReceiver, BurstStrategy
        >>> receiver = FIFOReceiver(strategy=BurstStrategy(burst_len=4, pause_cycles=2))
        >>> receiver.connect(dut=dut, clk=dut.clk,
        ...                  data=dut.dout, valid=dut.dout_valid,
        ...                  ready=dut.dout_ready)
        >>> stop = Event()
        >>> cocotb.start_soon(receiver.run(stop))
    """

    def __init__(self, strategy: TransferStrategy = None, debug: bool = False):
        """Initialize a FIFOReceiver.

        Args:
            strategy: Transfer strategy (defaults to GreedyStrategy).
            debug: If True, enable debug logging for each captured value.
        """
        self.strategy = strategy or GreedyStrategy()
        self.debug = debug
        self.data: LogicSequence = LogicSequence()
        self.timestamps: List[tuple] = []
        self.tag = None

    @classmethod
    def from_path(cls, path: str, dut, clk, strategy=None, debug=False):
        """Create a FIFOReceiver by resolving signals from ComponentDB.

        Args:
            path: ComponentDB path (e.g., ``"top.decoder.output"``).
            dut: DUT object.
            clk: Clock signal.
            strategy: TransferStrategy (defaults to GreedyStrategy).
            debug: Enable debug logging.

        Returns:
            A connected FIFOReceiver instance.
        """
        from .component_db import ComponentDB
        signals = ComponentDB.get(path)
        receiver = cls(strategy=strategy, debug=debug)
        receiver.tag = path.split(".")[-1]
        receiver.connect(
            dut=dut, clk=clk,
            data=signals["data"],
            valid=signals["valid"],
            ready=signals["ready"],
        )
        ComponentDB.register_component(receiver)
        return receiver

    def connect(self, dut, clk, data, valid, ready) -> None:
        """Connect the FIFOReceiver to HDL signals.

        Args:
            dut: The DUT object.
            clk: Clock signal.
            data: Data input signal (from DUT).
            valid: Valid input signal (from DUT).
            ready: Ready output signal (driven by receiver).
        """
        self.dut = dut
        self.clk = clk
        self.data_port = InputPort(data)
        self.valid_port = InputPort(valid)
        self.ready_port = OutputPort(ready)

    def dump(self) -> LogicSequence:
        """Get all collected data values.

        Returns:
            LogicSequence of captured data values in order.
        """
        return self.data

    def dump_time(self) -> List[tuple]:
        """Get timestamps for each collected value.

        Returns:
            List of simulation time values for each captured value.
        """
        return self.timestamps

    async def run(self, stop_event) -> None:
        """Start capturing data.

        Asserts ``ready`` based on the strategy, and captures data
        whenever both ``valid`` and ``ready`` are high.

        Args:
            stop_event: cocotb Event that signals when to stop.
        """
        from cocotb.triggers import RisingEdge
        from cocotb.utils import get_sim_time

        self.strategy.reset()
        cycle = 0
        while not stop_event.is_set():
            want = self.strategy.should_transfer(cycle)
            self.ready_port.set(1 if want else 0)

            await RisingEdge(self.clk)
            cycle += 1

            v_val = self.valid_port.get()
            is_valid = (v_val == 1) and want

            if is_valid:
                val = self.data_port.get()
                t = get_sim_time()
                self.data.append(val)
                self.timestamps.append(t)
                if self.debug:
                    self.dut._log.info(f"[FIFOReceiver] Captured {val} at {t}")
