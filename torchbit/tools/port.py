"""
Signal port wrapper abstractions for Cocotb.

Provides InputPort and OutputPort classes that wrap Cocotb signals
with a unified interface for reading and writing. Handles None signals
gracefully and provides optional immediate value setting.
"""
from typing import Optional


class InputPort:
    """Wrapper for reading HDL signal values.

    Provides a unified interface for reading from Cocotb signals,
    handling None signals gracefully (returns 0 for None).

    Example:
        >>> port = InputPort(dut.data_in)
        >>> value = port.get()  # Returns int value of the signal
        >>> # If wrapper is None, returns 0
    """

    def __init__(self, wrapper_signal):
        """Initialize an InputPort.

        Args:
            wrapper_signal: Cocotb signal handle or None.
        """
        self.wrapper = wrapper_signal

    def get(self) -> int:
        """Read the current signal value.

        Returns:
            Integer value of the signal, or 0 if wrapper is None.

        Raises:
            AttributeError: If wrapper is not None but lacks .value attribute.
        """
        if self.wrapper is None:
            return 0
        return int(self.wrapper.value)


class OutputPort:
    """Wrapper for driving HDL signal values.

    Provides a unified interface for driving Cocotb signals,
    with optional immediate value setting for time-zero initialization.

    Example:
        >>> port = OutputPort(dut.data_out, set_immediately=True)
        >>> port.set(0xDEAD)  # Drive the signal
        >>>
        >>> # Without set_immediately (default):
        >>> port = OutputPort(dut.data_out)
        >>> port.set(0xBEEF)  # Drive on next delta cycle
    """

    def __init__(self, wrapper_signal, set_immediately: bool = False):
        """Initialize an OutputPort.

        Args:
            wrapper_signal: Cocotb signal handle or None.
            set_immediately: If True, use setimmediatevalue() instead of
                           value assignment. Useful for time-zero initialization.
        """
        self.wrapper = wrapper_signal
        self.set_immediately = set_immediately

    def set(self, value: int) -> None:
        """Drive the signal to the specified value.

        Args:
            value: Integer value to drive onto the signal.

        Note:
            If wrapper is None, this is a no-op.
            If set_immediately is True, uses setimmediatevalue() for
            immediate effect without waiting for a delta cycle.
        """
        if self.wrapper is None:
            return
        if self.set_immediately:
            self.wrapper.setimmediatevalue(value)
        else:
            self.wrapper.value = value
