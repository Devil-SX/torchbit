"""
Configuration dataclasses for simulator builds.

Provides FileConfig, BuildConfig, VerilatorBuildArgs, and VCSBuildArgs
for configuring HDL source files and simulator build options.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FileConfig:
    """Configuration for HDL source files and top-level module.

    Attributes:
        name (str): Identifier for this configuration.
        sources (list[Path]): List of HDL source file paths.
        top_design (str): Name of the top-level module/entity.
        includes (list[Path]): Optional include directories for Verilog `include.

    Example:
        >>> from pathlib import Path
        >>> config = FileConfig(
        ...     name="my_design",
        ...     sources=[Path("top.sv"), Path("submodule.sv")],
        ...     top_design="top",
        ...     includes=[Path("includes")]
        ... )
    """
    name: str
    # Relative paths to the source files
    sources: List[Path | str]
    top_design: str
    includes: List[Path | str] = field(default_factory=list)

    def __post_init__(self):
        self.sources = [Path(src) for src in self.sources]
        self.includes = [inc for inc in self.includes]


@dataclass
class VerilatorBuildArgs:
    """Pre-configured build arguments for Verilator.

    Sets up common Verilator compilation options including:
    - Multi-threading (compiler and runtime)
    - Warning suppression for common warnings
    - FST waveform tracing

    Attributes:
        compiler_threads (int): Parallel compilation threads. Default: 16.
        runtime_threads (int): Verilator runtime threads. Default: 4.
        wno_timescalemod (bool): Suppress TIMESCALEMOD warnings. Default: True.
        wno_widthexpand (bool): Suppress WIDTHEXPAND warnings. Default: True.
        wno_widthtrunc (bool): Suppress WIDTHTRUNC warnings. Default: True.
        wno_unoptflat (bool): Suppress UNOPTFLAT warnings. Default: True.
        wno_unsigned (bool): Suppress UNSIGNED warnings. Default: True.
        wno_casex (bool): Suppress CASEX warnings. Default: True.
        wno_ascrange (bool): Suppress ASCRANGE warnings. Default: True.
        wno_latch (bool): Suppress LATCH warnings. Default: True.
        wno_moddup (bool): Suppress MODDUP warnings. Default: True.
        trace (bool): Enable waveform tracing (FST). Default: True.

    Example:
        >>> # Custom configuration
        >>> args = VerilatorBuildArgs(
        ...     compiler_threads=8,
        ...     runtime_threads=2,
        ...     trace=False
        ... )
        >>> build_args = args.get_build_args()
    """
    compiler_threads: int = 16
    runtime_threads: int = 4
    wno_timescalemod: bool = True
    wno_widthexpand: bool = True
    wno_widthtrunc: bool = True
    wno_unoptflat: bool = True
    wno_unsigned: bool = True
    wno_casex: bool = True
    wno_ascrange: bool = True
    wno_latch: bool = True
    wno_moddup: bool = True
    trace: bool = True

    def get_build_args(self) -> List[str]:
        """Generate the list of build arguments.

        Returns:
            List of command-line arguments for Verilator.
        """
        build_args = []
        if self.wno_timescalemod:
            build_args += ["-Wno-TIMESCALEMOD"]
        if self.wno_widthexpand:
            build_args += ["-Wno-WIDTHEXPAND"]
        if self.wno_unoptflat:
            build_args += ["-Wno-UNOPTFLAT"]
        if self.wno_widthtrunc:
            build_args += ["-Wno-WIDTHTRUNC"]
        if self.wno_unsigned:
            build_args += ["-Wno-UNSIGNED"]
        if self.wno_casex:
            build_args += ["-Wno-CASEX"]
        if self.wno_ascrange:
            build_args += ["-Wno-ASCRANGE"]
        if self.wno_latch:
            build_args += ["-Wno-LATCH"]
        if self.wno_moddup:
            build_args += ["-Wno-MODDUP"]

        # Suppress PINMISSING warnings for unconnected afifo ports
        build_args += ["-Wno-PINMISSING"]

        build_args += ["-j", str(self.compiler_threads)]
        build_args += ["--threads", str(self.runtime_threads)]
        if self.trace:
            build_args += ["--trace", "--trace-fst"]
        build_args += ["--timing"]  # Added to support timing delays in afifo.sv
        return build_args


@dataclass
class VCSBuildArgs:
    """Pre-configured build arguments for Synopsys VCS.

    Sets up common VCS compilation options including:
    - Timescale configuration
    - Debug database (kdb)
    - Timing checks disable for faster simulation

    Note:
        Includes +notimingcheck and +nospecify to speed up simulation
        and ignore PDK library timing models.

    Example:
        >>> args = VCSBuildArgs()
        >>> build_args = args.get_build_args()
    """

    def get_build_args(self) -> List[str]:
        """Generate the list of build arguments.

        Returns:
            List of command-line arguments for VCS.
        """
        build_args = []
        # build_args += ["-R"] # Run after build, not needed here
        build_args += ["+notimingcheck"]
        build_args += ["+nospecify"]  # Ignore specify blocks, which are used in PDK libraries mostly for modeling pin delays
        # build_args += ["+v2k"] # Verilog 2001 Standard
        build_args += ["-timescale=1ns/1ps"]  # Would override any `timescale directives in the source files, otherwise the vcs would check for consistency between all timescale directives in the source files
        build_args += ["-kdb"]  # Enable VCS debug database
        return build_args


@dataclass
class BuildConfig:
    """Simulator build configuration.

    Combines backend selection with build arguments.

    Attributes:
        name (str): Configuration name. Default: "default_verilator".
        backend (str): Simulator backend ("verilator" or "vcs").
        build_args (list): Additional build arguments.
        simulator_build_args (VerilatorBuildArgs or VCSBuildArgs): Backend-specific args.

    Raises:
        AssertionError: If backend is not "vcs" or "verilator".

    Example:
        >>> # Use default Verilator configuration
        >>> config = BuildConfig()
        >>>
        >>> # Use default VCS configuration
        >>> vcs_config = BuildConfig(name="default_vcs", backend="vcs")
        >>>
        >>> # Custom configuration
        >>> custom = BuildConfig(
        ...     name="custom",
        ...     backend="verilator",
        ...     build_args=["-O3", "-march=native"]
        ... )
    """
    name: str = "default_verilator"
    backend: str = "verilator"
    build_args: List[str] = None
    simulator_build_args: VerilatorBuildArgs | VCSBuildArgs = None

    def __post_init__(self):
        assert self.backend in ["vcs", "verilator"], "Unsupported backend specified"

        if self.build_args is None:
            self.build_args = []

        if self.simulator_build_args is None:
            if self.backend == "verilator":
                self.simulator_build_args = VerilatorBuildArgs()
            elif self.backend == "vcs":
                self.simulator_build_args = VCSBuildArgs()

        if self.simulator_build_args is not None:
            self.build_args += self.simulator_build_args.get_build_args()


DEFAULT_VERILATOR_BUILD_CONFIG = BuildConfig()
DEFAULT_VCS_BUILD_CONFIG = BuildConfig(name="default_vcs", backend="vcs")
