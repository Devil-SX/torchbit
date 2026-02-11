"""pyuvm-compatible agent assembling driver + monitor + sequencer."""


class TorchbitAgent:
    """pyuvm-compatible agent (documentation-level without pyuvm).

    Assembles a TorchbitDriver, TorchbitMonitor, and sequencer.
    """

    pass


def create_uvm_agent(
    driver_cls=None, monitor_cls=None, name: str = "TorchbitUvmAgent"
):
    """Factory that creates a uvm_agent subclass.

    Args:
        driver_cls: uvm_driver subclass (from create_uvm_driver).
        monitor_cls: uvm_monitor subclass (from create_uvm_monitor).
        name: Class name for the generated agent.

    Returns:
        A class inheriting from pyuvm.uvm_agent.
    """
    try:
        from pyuvm import uvm_agent, uvm_sequencer
    except ImportError:
        raise ImportError(
            "torchbit.uvm requires pyuvm. Install with: pip install pyuvm>=4.0.0"
        )

    if driver_cls is None:
        from .driver import create_uvm_driver

        driver_cls = create_uvm_driver()
    if monitor_cls is None:
        from .monitor import create_uvm_monitor

        monitor_cls = create_uvm_monitor()

    class _TorchbitUvmAgent(uvm_agent):
        def build_phase(self):
            super().build_phase()
            self.seqr = uvm_sequencer("seqr", self)
            self.drv = driver_cls.create("drv", self)
            self.mon = monitor_cls("mon", self)

        def connect_phase(self):
            self.drv.seq_item_port.connect(self.seqr.seq_item_export)

    _TorchbitUvmAgent.__name__ = name
    _TorchbitUvmAgent.__qualname__ = name
    return _TorchbitUvmAgent
