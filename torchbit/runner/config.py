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
class BuildConfig:
    name:str = "default_verilator"
    backend: str = "verilator"
    compiler_threads: int = 16
    runtime_threads: int = 4
    build_args: list = None
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

    def __post_init__(self):
        assert self.backend in ["verilator"], "Unsupported backend specified"
        if self.build_args is None:
            self.build_args = []
        if self.wno_timescalemod:
            self.build_args += ["-Wno-TIMESCALEMOD"]
        if self.wno_widthexpand:
            self.build_args += ["-Wno-WIDTHEXPAND"]
        if self.wno_unoptflat:
            self.build_args += ["-Wno-UNOPTFLAT"]
        if self.wno_widthtrunc:
            self.build_args += ["-Wno-WIDTHTRUNC"]
        if self.wno_unsigned:
            self.build_args += ["-Wno-UNSIGNED"]
        if self.wno_casex:
            self.build_args += ["-Wno-CASEX"]
        if self.wno_ascrange:
            self.build_args += ["-Wno-ASCRANGE"]
        if self.wno_latch:
            self.build_args += ["-Wno-LATCH"]
        if self.wno_moddup:
            self.build_args += ["-Wno-MODDUP"]
        
        # Suppress PINMISSING warnings for unconnected afifo ports
        self.build_args += ["-Wno-PINMISSING"]

        self.build_args += ["-j", str(self.compiler_threads)]
        self.build_args += ["--threads", str(self.runtime_threads)]
        if self.trace:
            self.build_args += ["--trace", "--trace-fst", "--trace-structs"]
        self.build_args += ["--timing"] # Added to support timing delays in afifo.sv

            



DEFAULT_BUILD_CONFIG = BuildConfig()