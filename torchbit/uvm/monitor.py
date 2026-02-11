"""pyuvm-compatible monitor using torchbit BFM."""


class TorchbitMonitor:
    """pyuvm-compatible monitor with analysis port.

    Without pyuvm installed, serves as documentation of the interface.
    """

    def __init__(self, bfm, debug: bool = False):
        self.bfm = bfm
        self.debug = debug


def create_uvm_monitor(name: str = "TorchbitUvmMonitor"):
    """Factory that creates a uvm_monitor subclass using torchbit BFM.

    Requires pyuvm to be installed.
    """
    try:
        from pyuvm import uvm_monitor, uvm_analysis_port, ConfigDB
    except ImportError:
        raise ImportError(
            "torchbit.uvm requires pyuvm. Install with: pip install pyuvm>=4.0.0"
        )
    from ..core.logic_sequence import LogicSequence

    class _TorchbitUvmMonitor(uvm_monitor):
        def build_phase(self):
            super().build_phase()
            self.ap = uvm_analysis_port("ap", self)
            self.bfm = ConfigDB().get(self, "", "BFM")

        async def run_phase(self):
            from cocotb.triggers import RisingEdge
            from .seq_items import VectorItem
            from ..core.vector import Vector

            self.raise_objection()
            while True:
                await RisingEdge(self.bfm.clk)
                if self.bfm.has_output("valid") and self.bfm.sample("valid"):
                    val = self.bfm.sample("data")
                    item = VectorItem("captured", Vector.from_int(val, 1, None))
                    self.ap.write(item)
            self.drop_objection()

    _TorchbitUvmMonitor.__name__ = name
    _TorchbitUvmMonitor.__qualname__ = name
    return _TorchbitUvmMonitor
