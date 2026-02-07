"""
Tests for Driver, PoolMonitor, and FIFOMonitor classes.

These classes are used for driving and collecting data in Cocotb testbenches.
"""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from cocotb.triggers import RisingEdge
from cocotb.utils import get_sim_time

from torchbit.tools.driver import Driver
from torchbit.tools.monitor import PoolMonitor, FIFOMonitor
from torchbit.tools.port import InputPort, OutputPort


class MockClock:
    """A mock clock signal for testing."""

    def __init__(self):
        self.edge_count = 0


class MockDUT:
    """A mock DUT object for testing."""

    def __init__(self):
        self.clk = MockClock()
        self._log = Mock()
        self.data_in = MockSignal(0)
        self.data_out = MockSignal(0)
        self.valid_in = MockSignal(0)
        self.valid_out = MockSignal(0)
        self.full = MockSignal(0)
        self.ready = MockSignal(0)
        self.empty = MockSignal(0)

    def log_info(self, msg):
        if hasattr(self._log, 'info'):
            self._log.info(msg)


class MockSignal:
    """A mock signal class that mimics cocotb signal behavior."""

    def __init__(self, value):
        self._value = value
        self.value = value

    def __int__(self):
        return int(self._value)


class TestDriver:
    """Tests for the Driver class."""

    def test_driver_init_default(self):
        """Test Driver initialization with default parameters."""
        driver = Driver()
        assert driver.debug is False
        assert driver.queue == []
        assert driver.timestamps == []

    def test_driver_init_with_debug(self):
        """Test Driver initialization with debug=True."""
        driver = Driver(debug=True)
        assert driver.debug is True

    def test_driver_load(self):
        """Test Driver.load() loads a sequence."""
        driver = Driver()
        sequence = [1, 2, 3, 4]
        driver.load(sequence)

        assert driver.queue == sequence
        assert driver.timestamps == []

    def test_driver_load_clears_previous_data(self):
        """Test Driver.load() clears previous queue and timestamps."""
        driver = Driver()
        driver.load([1, 2, 3])
        driver.timestamps = [1, 2, 3]

        driver.load([4, 5, 6])
        assert driver.queue == [4, 5, 6]
        assert driver.timestamps == []

    def test_driver_connect(self):
        """Test Driver.connect() sets up port connections."""
        driver = Driver()
        dut = MockDUT()

        driver.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)

        assert driver.dut is dut
        assert driver.clk is dut.clk
        assert isinstance(driver.data_port, OutputPort)
        assert isinstance(driver.valid_port, OutputPort)

    def test_driver_connect_with_full_signal(self):
        """Test Driver.connect() with full (backpressure) signal."""
        driver = Driver()
        dut = MockDUT()

        driver.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in, full=dut.full)

        assert driver.full_port is not None
        assert isinstance(driver.full_port, InputPort)

    def test_driver_connect_without_full_signal(self):
        """Test Driver.connect() without full signal."""
        driver = Driver()
        dut = MockDUT()

        driver.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)

        assert driver.full_port is None

    def test_driver_dump_time(self):
        """Test Driver.dump_time() returns timestamps."""
        driver = Driver()
        driver.timestamps = [(1, 'ns'), (2, 'ns'), (3, 'ns')]

        assert driver.dump_time() == [(1, 'ns'), (2, 'ns'), (3, 'ns')]

    def test_driver_load_empty_sequence(self):
        """Test Driver.load() with empty sequence."""
        driver = Driver()
        driver.load([])
        assert driver.queue == []


class TestPoolMonitor:
    """Tests for the PoolMonitor class."""

    def test_pool_monitor_init_default(self):
        """Test PoolMonitor initialization with default parameters."""
        monitor = PoolMonitor()
        assert monitor.debug is False
        assert monitor.data == []
        assert monitor.timestamps == []

    def test_pool_monitor_init_with_debug(self):
        """Test PoolMonitor initialization with debug=True."""
        monitor = PoolMonitor(debug=True)
        assert monitor.debug is True

    def test_pool_monitor_connect(self):
        """Test PoolMonitor.connect() sets up port connections."""
        monitor = PoolMonitor()
        dut = MockDUT()

        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out, valid=dut.valid_out)

        assert monitor.dut is dut
        assert monitor.clk is dut.clk
        assert isinstance(monitor.data_port, InputPort)
        assert isinstance(monitor.valid_port, InputPort)

    def test_pool_monitor_dump(self):
        """Test PoolMonitor.dump() returns collected data."""
        monitor = PoolMonitor()
        monitor.data = [1, 2, 3, 4, 5]

        assert monitor.dump() == [1, 2, 3, 4, 5]

    def test_pool_monitor_dump_time(self):
        """Test PoolMonitor.dump_time() returns timestamps."""
        monitor = PoolMonitor()
        monitor.timestamps = [(1, 'ns'), (2, 'ns'), (3, 'ns')]

        assert monitor.dump_time() == [(1, 'ns'), (2, 'ns'), (3, 'ns')]

    def test_pool_monitor_dump_empty(self):
        """Test PoolMonitor.dump() with no data collected."""
        monitor = PoolMonitor()
        assert monitor.dump() == []


class TestFIFOMonitor:
    """Tests for the FIFOMonitor class."""

    def test_fifo_monitor_init_default(self):
        """Test FIFOMonitor initialization with default parameters."""
        monitor = FIFOMonitor()
        assert monitor.debug is False
        assert monitor.data == []
        assert monitor.timestamps == []

    def test_fifo_monitor_init_with_debug(self):
        """Test FIFOMonitor initialization with debug=True."""
        monitor = FIFOMonitor(debug=True)
        assert monitor.debug is True

    def test_fifo_monitor_connect(self):
        """Test FIFOMonitor.connect() sets up port connections."""
        monitor = FIFOMonitor()
        dut = MockDUT()

        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out, empty=dut.empty, ready=dut.ready)

        assert monitor.dut is dut
        assert monitor.clk is dut.clk
        assert isinstance(monitor.data_port, InputPort)
        assert isinstance(monitor.empty_port, InputPort)
        assert isinstance(monitor.ready_port, OutputPort)

    def test_fifo_monitor_connect_with_valid(self):
        """Test FIFOMonitor.connect() with valid signal."""
        monitor = FIFOMonitor()
        dut = MockDUT()

        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out, empty=dut.empty,
                         ready=dut.ready, valid=dut.valid_out)

        assert monitor.valid_port is not None
        assert isinstance(monitor.valid_port, InputPort)

    def test_fifo_monitor_connect_without_valid(self):
        """Test FIFOMonitor.connect() without valid signal."""
        monitor = FIFOMonitor()
        dut = MockDUT()

        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out, empty=dut.empty, ready=dut.ready)

        assert monitor.valid_port is None

    def test_fifo_monitor_dump(self):
        """Test FIFOMonitor.dump() returns collected data."""
        monitor = FIFOMonitor()
        monitor.data = [10, 20, 30]

        assert monitor.dump() == [10, 20, 30]

    def test_fifo_monitor_dump_time(self):
        """Test FIFOMonitor.dump_time() returns timestamps."""
        monitor = FIFOMonitor()
        monitor.timestamps = [(100, 'ps'), (200, 'ps')]

        assert monitor.dump_time() == [(100, 'ps'), (200, 'ps')]


class TestDriverMonitorIntegration:
    """Integration tests for Driver and Monitor classes."""

    def test_driver_and_pool_monitor_setup(self):
        """Test setting up both Driver and PoolMonitor with same DUT."""
        dut = MockDUT()

        driver = Driver(debug=False)
        driver.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)
        driver.load([1, 2, 3, 4])

        monitor = PoolMonitor(debug=False)
        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out, valid=dut.valid_out)

        assert len(driver.queue) == 4
        assert len(monitor.data) == 0

    def test_driver_and_fifo_monitor_setup(self):
        """Test setting up Driver and FIFOMonitor with same DUT."""
        dut = MockDUT()

        driver = Driver()
        driver.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in, full=dut.full)
        driver.load([5, 6, 7])

        monitor = FIFOMonitor()
        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out,
                         empty=dut.empty, ready=dut.ready)

        assert len(driver.queue) == 3
        assert monitor.data == []


class TestFlowControl:
    """Tests for flow control mechanisms."""

    def test_driver_with_none_full_port(self):
        """Test Driver behavior when full_port is None (no flow control)."""
        driver = Driver()
        dut = MockDUT()

        driver.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)

        # When full_port is None, can_send should always be True
        assert driver.full_port is None

    def test_fifo_monitor_empty_signal_handling(self):
        """Test FIFOMonitor empty signal connection."""
        monitor = FIFOMonitor()
        dut = MockDUT()

        monitor.connect(dut=dut, clk=dut.clk, data=dut.data_out,
                         empty=dut.empty, ready=dut.ready)

        # Should be able to access empty_port
        assert monitor.empty_port is not None
        assert isinstance(monitor.empty_port, InputPort)


class TestDebugLogging:
    """Tests for debug logging functionality."""

    def test_driver_debug_flag(self):
        """Test Driver debug flag is stored correctly."""
        driver_no_debug = Driver(debug=False)
        driver_with_debug = Driver(debug=True)

        assert driver_no_debug.debug is False
        assert driver_with_debug.debug is True

    def test_pool_monitor_debug_flag(self):
        """Test PoolMonitor debug flag is stored correctly."""
        monitor_no_debug = PoolMonitor(debug=False)
        monitor_with_debug = PoolMonitor(debug=True)

        assert monitor_no_debug.debug is False
        assert monitor_with_debug.debug is True

    def test_fifo_monitor_debug_flag(self):
        """Test FIFOMonitor debug flag is stored correctly."""
        monitor_no_debug = FIFOMonitor(debug=False)
        monitor_with_debug = FIFOMonitor(debug=True)

        assert monitor_no_debug.debug is False
        assert monitor_with_debug.debug is True
