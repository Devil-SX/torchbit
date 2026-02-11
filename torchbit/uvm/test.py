"""pyuvm-compatible test template."""


class TorchbitTest:
    """Documentation-level test (works without pyuvm).

    Orchestrates environment setup and stimulus generation.
    """

    def __init__(self, name: str = "test"):
        self.name = name
        self.env = None


def create_uvm_test(env_cls=None, name: str = "TorchbitUvmTest"):
    """Factory that creates a uvm_test subclass.

    Args:
        env_cls: uvm_env subclass (from create_uvm_env).
        name: Class name for the generated test.

    Returns:
        A class inheriting from pyuvm.uvm_test.
    """
    try:
        from pyuvm import uvm_test
    except ImportError:
        raise ImportError(
            "torchbit.uvm requires pyuvm. Install with: pip install pyuvm>=4.0.0"
        )
    if env_cls is None:
        from .env import create_uvm_env

        env_cls = create_uvm_env()

    class _TorchbitUvmTest(uvm_test):
        def build_phase(self):
            super().build_phase()
            self.env = env_cls.create("env", self)

    _TorchbitUvmTest.__name__ = name
    _TorchbitUvmTest.__qualname__ = name
    return _TorchbitUvmTest
