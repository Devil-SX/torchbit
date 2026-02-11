"""BFM adapter bridging torchbit ports and pyuvm components."""
from ..tools.port import InputPort, OutputPort


class TorchbitBFM:
    """Bridge between pyuvm components and cocotb DUT signals.

    Wraps InputPort/OutputPort for use with pyuvm's ConfigDB pattern.
    The BFM itself is pure Python — cocotb is only needed when
    actually driving/sampling signals at simulation time.
    """

    def __init__(self, dut, clk):
        self.dut = dut
        self.clk = clk
        self._inputs = {}
        self._outputs = {}

    def add_input(self, name: str, signal) -> None:
        """Register a DUT input signal (driven by testbench)."""
        self._inputs[name] = OutputPort(signal)  # TB drives -> OutputPort

    def add_output(self, name: str, signal) -> None:
        """Register a DUT output signal (sampled by testbench)."""
        self._outputs[name] = InputPort(signal)  # TB reads -> InputPort

    def drive(self, name: str, value: int) -> None:
        """Drive a registered input signal."""
        self._inputs[name].set(value)

    def sample(self, name: str) -> int:
        """Sample a registered output signal."""
        return self._outputs[name].get()

    def has_input(self, name: str) -> bool:
        return name in self._inputs

    def has_output(self, name: str) -> bool:
        return name in self._outputs
