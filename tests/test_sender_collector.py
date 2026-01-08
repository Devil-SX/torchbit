"""
Tests for Sender, PoolCollector, and FIFOCollector classes.

These classes are used for driving and collecting data in Cocotb testbenches.
"""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from cocotb.triggers import RisingEdge
from cocotb.utils import get_sim_time

from torchbit.tools.sender_collector import Sender, PoolCollector, FIFOCollector
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


class TestSender:
    """Tests for the Sender class."""

    def test_sender_init_default(self):
        """Test Sender initialization with default parameters."""
        sender = Sender()
        assert sender.debug is False
        assert sender.queue == []
        assert sender.timestamps == []

    def test_sender_init_with_debug(self):
        """Test Sender initialization with debug=True."""
        sender = Sender(debug=True)
        assert sender.debug is True

    def test_sender_load(self):
        """Test Sender.load() loads a sequence."""
        sender = Sender()
        sequence = [1, 2, 3, 4]
        sender.load(sequence)

        assert sender.queue == sequence
        assert sender.timestamps == []

    def test_sender_load_clears_previous_data(self):
        """Test Sender.load() clears previous queue and timestamps."""
        sender = Sender()
        sender.load([1, 2, 3])
        sender.timestamps = [1, 2, 3]

        sender.load([4, 5, 6])
        assert sender.queue == [4, 5, 6]
        assert sender.timestamps == []

    def test_sender_connect(self):
        """Test Sender.connect() sets up port connections."""
        sender = Sender()
        dut = MockDUT()

        sender.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)

        assert sender.dut is dut
        assert sender.clk is dut.clk
        assert isinstance(sender.data_port, OutputPort)
        assert isinstance(sender.valid_port, OutputPort)

    def test_sender_connect_with_full_signal(self):
        """Test Sender.connect() with full (backpressure) signal."""
        sender = Sender()
        dut = MockDUT()

        sender.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in, full=dut.full)

        assert sender.full_port is not None
        assert isinstance(sender.full_port, InputPort)

    def test_sender_connect_without_full_signal(self):
        """Test Sender.connect() without full signal."""
        sender = Sender()
        dut = MockDUT()

        sender.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)

        assert sender.full_port is None

    def test_sender_dump_time(self):
        """Test Sender.dump_time() returns timestamps."""
        sender = Sender()
        sender.timestamps = [(1, 'ns'), (2, 'ns'), (3, 'ns')]

        assert sender.dump_time() == [(1, 'ns'), (2, 'ns'), (3, 'ns')]

    def test_sender_load_empty_sequence(self):
        """Test Sender.load() with empty sequence."""
        sender = Sender()
        sender.load([])
        assert sender.queue == []


class TestPoolCollector:
    """Tests for the PoolCollector class."""

    def test_pool_collector_init_default(self):
        """Test PoolCollector initialization with default parameters."""
        collector = PoolCollector()
        assert collector.debug is False
        assert collector.data == []
        assert collector.timestamps == []

    def test_pool_collector_init_with_debug(self):
        """Test PoolCollector initialization with debug=True."""
        collector = PoolCollector(debug=True)
        assert collector.debug is True

    def test_pool_collector_connect(self):
        """Test PoolCollector.connect() sets up port connections."""
        collector = PoolCollector()
        dut = MockDUT()

        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out, valid=dut.valid_out)

        assert collector.dut is dut
        assert collector.clk is dut.clk
        assert isinstance(collector.data_port, InputPort)
        assert isinstance(collector.valid_port, InputPort)

    def test_pool_collector_dump(self):
        """Test PoolCollector.dump() returns collected data."""
        collector = PoolCollector()
        collector.data = [1, 2, 3, 4, 5]

        assert collector.dump() == [1, 2, 3, 4, 5]

    def test_pool_collector_dump_time(self):
        """Test PoolCollector.dump_time() returns timestamps."""
        collector = PoolCollector()
        collector.timestamps = [(1, 'ns'), (2, 'ns'), (3, 'ns')]

        assert collector.dump_time() == [(1, 'ns'), (2, 'ns'), (3, 'ns')]

    def test_pool_collector_dump_empty(self):
        """Test PoolCollector.dump() with no data collected."""
        collector = PoolCollector()
        assert collector.dump() == []


class TestFIFOCollector:
    """Tests for the FIFOCollector class."""

    def test_fifo_collector_init_default(self):
        """Test FIFOCollector initialization with default parameters."""
        collector = FIFOCollector()
        assert collector.debug is False
        assert collector.data == []
        assert collector.timestamps == []

    def test_fifo_collector_init_with_debug(self):
        """Test FIFOCollector initialization with debug=True."""
        collector = FIFOCollector(debug=True)
        assert collector.debug is True

    def test_fifo_collector_connect(self):
        """Test FIFOCollector.connect() sets up port connections."""
        collector = FIFOCollector()
        dut = MockDUT()

        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out, empty=dut.empty, ready=dut.ready)

        assert collector.dut is dut
        assert collector.clk is dut.clk
        assert isinstance(collector.data_port, InputPort)
        assert isinstance(collector.empty_port, InputPort)
        assert isinstance(collector.ready_port, OutputPort)

    def test_fifo_collector_connect_with_valid(self):
        """Test FIFOCollector.connect() with valid signal."""
        collector = FIFOCollector()
        dut = MockDUT()

        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out, empty=dut.empty,
                         ready=dut.ready, valid=dut.valid_out)

        assert collector.valid_port is not None
        assert isinstance(collector.valid_port, InputPort)

    def test_fifo_collector_connect_without_valid(self):
        """Test FIFOCollector.connect() without valid signal."""
        collector = FIFOCollector()
        dut = MockDUT()

        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out, empty=dut.empty, ready=dut.ready)

        assert collector.valid_port is None

    def test_fifo_collector_dump(self):
        """Test FIFOCollector.dump() returns collected data."""
        collector = FIFOCollector()
        collector.data = [10, 20, 30]

        assert collector.dump() == [10, 20, 30]

    def test_fifo_collector_dump_time(self):
        """Test FIFOCollector.dump_time() returns timestamps."""
        collector = FIFOCollector()
        collector.timestamps = [(100, 'ps'), (200, 'ps')]

        assert collector.dump_time() == [(100, 'ps'), (200, 'ps')]


class TestSenderCollectorIntegration:
    """Integration tests for Sender and Collector classes."""

    def test_sender_and_pool_collector_setup(self):
        """Test setting up both Sender and PoolCollector with same DUT."""
        dut = MockDUT()

        sender = Sender(debug=False)
        sender.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)
        sender.load([1, 2, 3, 4])

        collector = PoolCollector(debug=False)
        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out, valid=dut.valid_out)

        assert len(sender.queue) == 4
        assert len(collector.data) == 0

    def test_sender_and_fifo_collector_setup(self):
        """Test setting up Sender and FIFOCollector with same DUT."""
        dut = MockDUT()

        sender = Sender()
        sender.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in, full=dut.full)
        sender.load([5, 6, 7])

        collector = FIFOCollector()
        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out,
                         empty=dut.empty, ready=dut.ready)

        assert len(sender.queue) == 3
        assert collector.data == []


class TestFlowControl:
    """Tests for flow control mechanisms."""

    def test_sender_with_none_full_port(self):
        """Test Sender behavior when full_port is None (no flow control)."""
        sender = Sender()
        dut = MockDUT()

        sender.connect(dut=dut, clk=dut.clk, data=dut.data_in, valid=dut.valid_in)

        # When full_port is None, can_send should always be True
        assert sender.full_port is None

    def test_fifo_collector_empty_signal_handling(self):
        """Test FIFOCollector empty signal connection."""
        collector = FIFOCollector()
        dut = MockDUT()

        collector.connect(dut=dut, clk=dut.clk, data=dut.data_out,
                         empty=dut.empty, ready=dut.ready)

        # Should be able to access empty_port
        assert collector.empty_port is not None
        assert isinstance(collector.empty_port, InputPort)


class TestDebugLogging:
    """Tests for debug logging functionality."""

    def test_sender_debug_flag(self):
        """Test Sender debug flag is stored correctly."""
        sender_no_debug = Sender(debug=False)
        sender_with_debug = Sender(debug=True)

        assert sender_no_debug.debug is False
        assert sender_with_debug.debug is True

    def test_pool_collector_debug_flag(self):
        """Test PoolCollector debug flag is stored correctly."""
        collector_no_debug = PoolCollector(debug=False)
        collector_with_debug = PoolCollector(debug=True)

        assert collector_no_debug.debug is False
        assert collector_with_debug.debug is True

    def test_fifo_collector_debug_flag(self):
        """Test FIFOCollector debug flag is stored correctly."""
        collector_no_debug = FIFOCollector(debug=False)
        collector_with_debug = FIFOCollector(debug=True)

        assert collector_no_debug.debug is False
        assert collector_with_debug.debug is True
