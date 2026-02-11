from importlib.metadata import version as _version

__version__ = _version("torchbit")

try:
    import cocotb as _cocotb
    from packaging import version as _version_pkg
    if _version_pkg.parse(_cocotb.__version__) < _version_pkg.parse("2.0.0"):
        import warnings
        warnings.warn(
            "torchbit verification tools require cocotb >= 2.0.0, "
            f"found {_cocotb.__version__}. Driver/Monitor/Buffer may not work.",
            UserWarning,
            stacklevel=2,
        )
except ImportError:
    pass

from . import core
from . import debug
from . import tools
from . import utils
from . import runner
from . import uvm
