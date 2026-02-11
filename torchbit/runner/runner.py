"""
Cocotb simulator runner for Verilator and VCS.

Provides the Runner class for compiling and running HDL simulations
with waveform dumping support.
"""
from .config import *
from pathlib import Path


def generate_vcs_dump_wave(path: Path, top_module: str) -> None:
    """Generate a VCS FSDB waveform dump module.

    Creates a Verilog module that initializes FSDB waveform dumping
    for VCS simulations.

    Args:
        path: Path to write the dump module file.
        top_module: Name of the top-level DUT module.

    Note:
        The generated module uses $fsdbDumpfile and $fsdbDumpvars
        to create FSDB waveform output.
    """
    with open(path, "w") as f:
        f.write(f"module dump_fsdb;\ninitial begin\n$fsdbDumpfile(\"dump.fsdb\");\n$fsdbDumpvars(0, {top_module});\n $fsdbDumpvars(\"+all\");\n end\nendmodule")


class Runner:
    """Cocotb simulator runner for Verilator and VCS.

    A high-level wrapper around cocotb_tools.runner that handles:
    - Build configuration and compilation
    - Waveform dump generation (FST for Verilator, FSDB for VCS)
    - Test execution

    Attributes:
        file_config (FileConfig): Source file configuration.
        build_config (BuildConfig): Build configuration.
        sources (list[Path]): List of HDL source files.
        top_design (str): Top-level module name.
        includes (list[Path]): Include directories.
        build_args (list): Build arguments.
        backend (str): Simulator backend name.
        current_dir (Path): Working directory for build outputs.

    Example:
        >>> from torchbit.runner import (
        ...     Runner, FileConfig, BuildConfig,
        ...     DEFAULT_VERILATOR_BUILD_CONFIG, read_filelist
        ... )
        >>>
        >>> # Read source files
        >>> files = read_filelist("filelist.f", base_path=".")
        >>>
        >>> # Configure
        >>> fc = FileConfig(name="my_design", sources=files, top_design="top")
        >>> runner = Runner(fc, DEFAULT_VERILATOR_BUILD_CONFIG)
        >>>
        >>> # Run test
        >>> runner.test("my_test")  # Runs my_test.py
        >>>
        >>> # Build outputs are in: sim_my_design_default_verilator_my_test/

    Notes:
        - For VCS, a dump_wave.v module is automatically generated
          and added to sources
        - Waveform files are created in the build directory
        - Verilator produces FST files, VCS produces FSDB files
    """

    def __init__(self, file_config: FileConfig, build_config: BuildConfig, current_dir: Path = Path(".")):
        """Initialize a Runner.

        Args:
            file_config: Configuration for HDL source files and top module.
            build_config: Configuration for simulator build options.
            current_dir: Working directory for build outputs.
        """
        self.file_config = file_config
        self.build_config = build_config
        self.sources = file_config.sources
        self.top_design = file_config.top_design
        self.includes = file_config.includes
        self.build_args = build_config.build_args
        self.backend = build_config.backend
        self.current_dir = current_dir

    def test(self, test_module: str) -> None:
        """Build and run a test.

        Compiles the HDL sources and runs the specified test module.
        Creates a timestamped build directory with all outputs.

        Args:
            test_module: Name of the test module file (without .py).

        Note:
            After execution, build outputs are in:
            {current_dir}/sim_{name}_{backend}_{test_module}/

            For Verilator: waveform.fst in the build directory
            For VCS: dump.fsdb in the build directory
        """
        build_dir = self.current_dir / f"sim_{self.file_config.name}_{self.build_config.name}_{test_module}"

        if self.backend == "vcs":
            dump_wave_path = build_dir / "dump_wave.v"
            generate_vcs_dump_wave(dump_wave_path, self.top_design)
            self.sources.append(dump_wave_path)
            # self.top_design = "dump_fsdb"
            self.build_args += ["-top", "dump_fsdb"]  # set multi-top modules

        from cocotb_tools.runner import get_runner
        runner = get_runner(self.backend)
        runner.build(
            verilog_sources=self.sources,
            hdl_toplevel=self.top_design,
            includes=self.includes,
            verbose=True,
            waves=True,
            build_args=self.build_args,
            build_dir=build_dir,
        )
        runner.test(hdl_toplevel=self.top_design, test_module=test_module, waves=True, verbose=True)
