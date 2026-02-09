from importlib.metadata import version as _version

__version__ = _version("torchbit")

import cocotb
from packaging import version

assert version.parse(cocotb.__version__) >= version.parse("2.0.0"), "Cocotb version must be == 2.x for torchbit >= 2.x, please upgrade cocotb."

from . import core
from . import debug
from . import tools
from . import utils
from . import runner
