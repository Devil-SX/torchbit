from dataclasses import dataclass, field
from pathlib import Path



@dataclass
class FileConfig:
    name: str
    # Relative paths to the source files
    sources: list[Path|str] 
    top_design: str
    includes: list[Path|str] = field(default_factory=list)

    def __post_init__(self):
        self.sources = [ Path(src) for src in self.sources]
        self.includes = [ inc for inc in self.includes]


@dataclass
class VerilatorBuildArgs:
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

    def get_build_args(self):
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
        build_args += ["--timing"] # Added to support timing delays in afifo.sv
        return build_args




@dataclass
class VCSBuildArgs:
    def get_build_args(self):
        build_args = []
        # build_args += ["-R"] # Run after build, not needed here
        build_args += ["+notimingcheck"]
        build_args += ["+nospecify"] # Ignore specify blocks, which are used in PDK libraries mostly for modeling pin delays
        # build_args += ["+v2k"] # Verilog 2001 Standard
        build_args += ["-timescale=1ns/1ps"] # Would override any `timescale directives in the source files, otherwise the vcs would check for consistency between all timescale directives in the source files
        build_args += ["-kdb"] # Enable VCS debug database
        return build_args



@dataclass
class BuildConfig:
    name:str = "default_verilator"
    backend: str = "verilator"
    build_args: list = None
    simulator_build_args: VerilatorBuildArgs = None


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