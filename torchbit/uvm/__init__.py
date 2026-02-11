"""pyuvm integration layer for torchbit.

Provides UVM-compatible components that wrap torchbit's data types
and verification components. Requires pyuvm >= 4.0.0.

All pyuvm imports are lazy — this package can be imported for
inspection even without pyuvm installed, but instantiating
components requires pyuvm.
"""
from .bfm import TorchbitBFM
from .seq_items import VectorItem, LogicSequenceItem
from .driver import TorchbitDriver
from .monitor import TorchbitMonitor
from .agent import TorchbitAgent
from .scoreboard import TorchbitScoreboard
from .env import TorchbitEnv
from .test import TorchbitTest
from .factory import ComponentRegistry
from .coverage import CoverageGroup, CoveragePoint
from .ral import RegisterModel, RegisterBlock
