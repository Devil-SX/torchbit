"""pyuvm-compatible driver using torchbit BFM."""
from ..tools.strategy import TransferStrategy, GreedyStrategy


class TorchbitDriver:
    """pyuvm-compatible driver that uses torchbit BFM for signal access.

    When pyuvm is installed, call ``create_uvm_driver()`` to get a class
    that inherits from uvm_driver. Without pyuvm, this serves as
    documentation of the interface.
    """

    def __init__(self, bfm, strategy: TransferStrategy = None, debug: bool = False):
        self.bfm = bfm
        self.strategy = strategy or GreedyStrategy()
        self.debug = debug


def create_uvm_driver(name: str = "TorchbitUvmDriver"):
    """Factory that creates a uvm_driver subclass using torchbit BFM.

    Requires pyuvm to be installed.

    Returns:
        A class inheriting from pyuvm.uvm_driver.
    """
    try:
        from pyuvm import uvm_driver, ConfigDB
    except ImportError:
        raise ImportError(
            "torchbit.uvm requires pyuvm. Install with: pip install pyuvm>=4.0.0"
        )
    from ..tools.strategy import GreedyStrategy

    class _TorchbitUvmDriver(uvm_driver):
        def build_phase(self):
            super().build_phase()
            self.bfm = ConfigDB().get(self, "", "BFM")
            self.strategy = ConfigDB().get(
                self, "", "STRATEGY", default=GreedyStrategy()
            )

        async def run_phase(self):
            from cocotb.triggers import RisingEdge

            self.raise_objection()
            cycle = 0
            while True:
                item = await self.seq_item_port.get_next_item()
                if item is None:
                    break

                while not self.strategy.should_transfer(cycle):
                    await RisingEdge(self.bfm.clk)
                    cycle += 1

                self.bfm.drive(
                    "data",
                    item.vector.to_logic()
                    if hasattr(item, "vector") and item.vector
                    else 0,
                )
                self.bfm.drive("valid", 1)
                await RisingEdge(self.bfm.clk)
                cycle += 1
                self.bfm.drive("valid", 0)

                self.seq_item_port.item_done()
            self.drop_objection()

    _TorchbitUvmDriver.__name__ = name
    _TorchbitUvmDriver.__qualname__ = name
    return _TorchbitUvmDriver
