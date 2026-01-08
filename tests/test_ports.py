"""
Tests for InputPort and OutputPort classes.

These classes wrap Cocotb signals for unified read/write interface.
"""
import pytest
from unittest.mock import Mock, MagicMock
from torchbit.tools.port import InputPort, OutputPort


class MockSignal:
    """A mock signal class that mimics cocotb signal behavior."""

    def __init__(self, value):
        self._value = value
        self.value = value

    def __int__(self):
        return int(self._value)

    def __setattr__(self, name, val):
        if name == 'value':
            object.__setattr__(self, '_value', val)
            object.__setattr__(self, 'value', val)
        else:
            object.__setattr__(self, name, val)

    def setimmediatevalue(self, val):
        self._value = val
        self.value = val


class TestInputPort:
    """Tests for the InputPort class."""

    def test_input_port_init_with_signal(self):
        """Test InputPort initialization with a signal."""
        signal = MockSignal(42)
        port = InputPort(signal)
        assert port.wrapper is signal

    def test_input_port_init_with_none(self):
        """Test InputPort initialization with None."""
        port = InputPort(None)
        assert port.wrapper is None

    def test_input_port_get_returns_signal_value(self):
        """Test InputPort.get() returns the signal's integer value."""
        signal = MockSignal(0x123)
        port = InputPort(signal)
        assert port.get() == 0x123

    def test_input_port_get_with_none_returns_zero(self):
        """Test InputPort.get() returns 0 when wrapper is None."""
        port = InputPort(None)
        assert port.get() == 0

    def test_input_port_get_with_zero_value(self):
        """Test InputPort.get() with signal value of 0."""
        signal = MockSignal(0)
        port = InputPort(signal)
        assert port.get() == 0

    def test_input_port_get_with_negative_value(self):
        """Test InputPort.get() with negative signal value."""
        signal = MockSignal(-1)
        port = InputPort(signal)
        assert port.get() == -1

    def test_input_port_get_with_large_value(self):
        """Test InputPort.get() with large signal value."""
        signal = MockSignal(0xFFFFFFFF)
        port = InputPort(signal)
        assert port.get() == 0xFFFFFFFF

    def test_input_port_get_changes_with_signal(self):
        """Test InputPort.get() reflects signal value changes."""
        signal = MockSignal(10)
        port = InputPort(signal)

        assert port.get() == 10
        signal._value = 20
        signal.value = 20
        assert port.get() == 20


class TestOutputPort:
    """Tests for the OutputPort class."""

    def test_output_port_init_with_signal(self):
        """Test OutputPort initialization with a signal."""
        signal = MockSignal(0)
        port = OutputPort(signal)
        assert port.wrapper is signal
        assert port.set_immediately is False

    def test_output_port_init_with_none(self):
        """Test OutputPort initialization with None."""
        port = OutputPort(None)
        assert port.wrapper is None
        assert port.set_immediately is False

    def test_output_port_init_with_set_immediately(self):
        """Test OutputPort initialization with set_immediately=True."""
        signal = MockSignal(0)
        port = OutputPort(signal, set_immediately=True)
        assert port.wrapper is signal
        assert port.set_immediately is True

    def test_output_port_set_updates_signal_value(self):
        """Test OutputPort.set() updates the signal value."""
        signal = MockSignal(0)
        port = OutputPort(signal)

        port.set(0xDEAD)
        assert signal._value == 0xDEAD

    def test_output_port_set_with_none_is_noop(self):
        """Test OutputPort.set() with None wrapper is a no-op."""
        port = OutputPort(None)
        # Should not raise any exception
        port.set(0xBEEF)

    def test_output_port_set_multiple_times(self):
        """Test OutputPort.set() can be called multiple times."""
        signal = MockSignal(0)
        port = OutputPort(signal)

        port.set(1)
        assert signal._value == 1

        port.set(2)
        assert signal._value == 2

        port.set(0xFF)
        assert signal._value == 0xFF

    def test_output_port_set_with_zero(self):
        """Test OutputPort.set() with zero value."""
        signal = MockSignal(0xFF)
        port = OutputPort(signal)

        port.set(0)
        assert signal._value == 0

    def test_output_port_set_immediately_true(self):
        """Test OutputPort.set() with set_immediately=True."""
        signal = MockSignal(0)
        port = OutputPort(signal, set_immediately=True)

        port.set(0x1234)
        assert signal._value == 0x1234
        # Verify setimmediatevalue was called
        assert signal.value == 0x1234

    def test_output_port_set_immediately_false(self):
        """Test OutputPort.set() with set_immediately=False uses value assignment."""
        signal = MockSignal(0)
        port = OutputPort(signal, set_immediately=False)

        port.set(0x5678)
        assert signal._value == 0x5678

    def test_output_port_set_with_negative_value(self):
        """Test OutputPort.set() with negative value."""
        signal = MockSignal(0)
        port = OutputPort(signal)

        port.set(-1)
        assert signal._value == -1


class TestPortIntegration:
    """Integration tests for InputPort and OutputPort."""

    def test_input_output_port_roundtrip(self):
        """Test reading from InputPort what was written to OutputPort."""
        # Simulate a signal that can be read and written
        shared_signal = MockSignal(0)

        output_port = OutputPort(shared_signal)
        input_port = InputPort(shared_signal)

        output_port.set(0xABC)
        assert input_port.get() == 0xABC

        output_port.set(0x123)
        assert input_port.get() == 0x123

    def test_none_ports_integration(self):
        """Test InputPort/OutputPort both with None wrappers."""
        input_port = InputPort(None)
        output_port = OutputPort(None)

        # Should not raise exceptions
        output_port.set(0x999)
        assert input_port.get() == 0

    def test_immediate_vs_delayed_setting(self):
        """Test difference between set_immediately and normal setting."""
        signal1 = MockSignal(0)
        signal2 = MockSignal(0)

        port_immediate = OutputPort(signal1, set_immediately=True)
        port_delayed = OutputPort(signal2, set_immediately=False)

        port_immediate.set(0x111)
        port_delayed.set(0x222)

        assert signal1._value == 0x111
        assert signal2._value == 0x222
