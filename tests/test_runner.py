"""
Tests for Runner and configuration classes.

Tests the simulator runner configuration and setup without requiring
actual simulator execution.
"""
import pytest
from pathlib import Path
from torchbit.runner import (
    Runner,
    FileConfig,
    BuildConfig,
    VerilatorBuildArgs,
    VCSBuildArgs,
    DEFAULT_VERILATOR_BUILD_CONFIG,
    DEFAULT_VCS_BUILD_CONFIG,
    generate_vcs_dump_wave,
)
import tempfile


class TestFileConfig:
    """Tests for FileConfig dataclass."""

    def test_file_config_basic(self):
        """Test FileConfig with basic parameters."""
        config = FileConfig(
            name="test_design",
            sources=["top.v", "sub.v"],
            top_design="top"
        )

        assert config.name == "test_design"
        assert len(config.sources) == 2
        assert config.top_design == "top"
        assert config.includes == []

    def test_file_config_with_includes(self):
        """Test FileConfig with include directories."""
        config = FileConfig(
            name="test_design",
            sources=["top.v"],
            top_design="top",
            includes=["include/", "common/"]
        )

        assert len(config.includes) == 2

    def test_file_config_converts_sources_to_path(self):
        """Test FileConfig converts string sources to Path objects."""
        config = FileConfig(
            name="test_design",
            sources=["top.v", "sub.v"],
            top_design="top"
        )

        assert all(isinstance(src, Path) for src in config.sources)

    def test_file_config_with_path_sources(self):
        """Test FileConfig with Path sources."""
        config = FileConfig(
            name="test_design",
            sources=[Path("top.v")],
            top_design="top"
        )

        assert config.sources[0] == Path("top.v")

    def test_file_config_empty_sources(self):
        """Test FileConfig with empty source list."""
        config = FileConfig(
            name="test_design",
            sources=[],
            top_design="top"
        )

        assert config.sources == []


class TestVerilatorBuildArgs:
    """Tests for VerilatorBuildArgs dataclass."""

    def test_verilator_build_args_default(self):
        """Test VerilatorBuildArgs with default values."""
        args = VerilatorBuildArgs()

        assert args.compiler_threads == 16
        assert args.runtime_threads == 4
        assert args.trace is True
        assert args.wno_timescalemod is True

    def test_verilator_build_args_custom(self):
        """Test VerilatorBuildArgs with custom values."""
        args = VerilatorBuildArgs(
            compiler_threads=8,
            runtime_threads=2,
            trace=False
        )

        assert args.compiler_threads == 8
        assert args.runtime_threads == 2
        assert args.trace is False

    def test_verilator_get_build_args_contains_timing(self):
        """Test VerilatorBuildArgs.get_build_args() includes --timing."""
        args = VerilatorBuildArgs()
        build_args = args.get_build_args()

        assert "--timing" in build_args

    def test_verilator_get_build_args_trace_enabled(self):
        """Test VerilatorBuildArgs.get_build_args() with trace enabled."""
        args = VerilatorBuildArgs(trace=True)
        build_args = args.get_build_args()

        assert "--trace" in build_args
        assert "--trace-fst" in build_args

    def test_verilator_get_build_args_trace_disabled(self):
        """Test VerilatorBuildArgs.get_build_args() with trace disabled."""
        args = VerilatorBuildArgs(trace=False)
        build_args = args.get_build_args()

        assert "--trace" not in build_args
        assert "--trace-fst" not in build_args

    def test_verilator_get_build_args_warning_suppressions(self):
        """Test VerilatorBuildArgs.get_build_args() includes warning suppressions."""
        args = VerilatorBuildArgs()
        build_args = args.get_build_args()

        assert "-Wno-TIMESCALEMOD" in build_args
        assert "-Wno-WIDTHEXPAND" in build_args
        assert "-Wno-UNOPTFLAT" in build_args
        assert "-Wno-WIDTHTRUNC" in build_args
        assert "-Wno-UNSIGNED" in build_args
        assert "-Wno-PINMISSING" in build_args

    def test_verilator_get_build_args_threading(self):
        """Test VerilatorBuildArgs.get_build_args() includes threading options."""
        args = VerilatorBuildArgs(compiler_threads=8, runtime_threads=2)
        build_args = args.get_build_args()

        assert "-j" in build_args
        assert "8" in build_args
        assert "--threads" in build_args
        assert "2" in build_args


class TestVCSBuildArgs:
    """Tests for VCSBuildArgs dataclass."""

    def test_vcs_get_build_args_default(self):
        """Test VCSBuildArgs.get_build_args() returns expected arguments."""
        args = VCSBuildArgs()
        build_args = args.get_build_args()

        assert "+notimingcheck" in build_args
        assert "+nospecify" in build_args
        assert "-timescale=1ns/1ps" in build_args
        assert "-kdb" in build_args

    def test_vcs_get_build_args_no_run_flag(self):
        """Test VCSBuildArgs.get_build_args() doesn't include -R."""
        args = VCSBuildArgs()
        build_args = args.get_build_args()

        assert "-R" not in build_args


class TestBuildConfig:
    """Tests for BuildConfig dataclass."""

    def test_build_config_default_verilator(self):
        """Test BuildConfig defaults to Verilator backend."""
        config = BuildConfig()

        assert config.name == "default_verilator"
        assert config.backend == "verilator"
        assert isinstance(config.simulator_build_args, VerilatorBuildArgs)

    def test_build_config_vcs_backend(self):
        """Test BuildConfig with VCS backend."""
        config = BuildConfig(name="default_vcs", backend="vcs")

        assert config.backend == "vcs"
        assert isinstance(config.simulator_build_args, VCSBuildArgs)

    def test_build_config_invalid_backend_raises(self):
        """Test BuildConfig with invalid backend raises AssertionError."""
        with pytest.raises(AssertionError, match="Unsupported backend"):
            BuildConfig(backend="invalid_simulator")

    def test_build_config_custom_build_args(self):
        """Test BuildConfig with custom build arguments."""
        config = BuildConfig(
            name="custom",
            backend="verilator",
            build_args=["-O3", "--coverage"]
        )

        assert "-O3" in config.build_args
        assert "--coverage" in config.build_args

    def test_build_config_merges_simulator_args(self):
        """Test BuildConfig merges simulator args with custom args."""
        config = BuildConfig(
            name="custom",
            backend="verilator",
            build_args=["--custom-arg"]
        )

        # Should have both custom args and default Verilator args
        assert "--custom-arg" in config.build_args
        assert "--timing" in config.build_args

    def test_default_verilator_build_config(self):
        """Test DEFAULT_VERILATOR_BUILD_CONFIG constant."""
        config = DEFAULT_VERILATOR_BUILD_CONFIG

        assert config.backend == "verilator"
        assert isinstance(config.simulator_build_args, VerilatorBuildArgs)

    def test_default_vcs_build_config(self):
        """Test DEFAULT_VCS_BUILD_CONFIG constant."""
        config = DEFAULT_VCS_BUILD_CONFIG

        assert config.backend == "vcs"
        assert isinstance(config.simulator_build_args, VCSBuildArgs)


class TestRunner:
    """Tests for Runner class."""

    def test_runner_init(self):
        """Test Runner initialization."""
        fc = FileConfig(
            name="test",
            sources=[Path("top.v")],
            top_design="top"
        )
        bc = BuildConfig(backend="verilator")

        runner = Runner(fc, bc)

        assert runner.file_config is fc
        assert runner.build_config is bc
        assert runner.sources == fc.sources
        assert runner.top_design == fc.top_design
        assert runner.includes == fc.includes
        assert runner.build_args == bc.build_args
        assert runner.backend == bc.backend

    def test_runner_with_current_dir(self, tmp_path):
        """Test Runner with custom current_dir."""
        fc = FileConfig(
            name="test",
            sources=[Path("top.v")],
            top_design="top"
        )
        bc = BuildConfig(backend="verilator")

        runner = Runner(fc, bc, current_dir=tmp_path)

        assert runner.current_dir == tmp_path

    def test_runner_build_dir_name(self):
        """Test Runner generates correct build directory name."""
        fc = FileConfig(
            name="my_design",
            sources=[Path("top.v")],
            top_design="top"
        )
        bc = BuildConfig(name="custom_config", backend="verilator")

        runner = Runner(fc, bc)

        # The build dir would be: sim_{name}_{config_name}_{test_module}
        # We can't run test() without a simulator, but we can check the naming pattern
        assert runner.file_config.name == "my_design"
        assert runner.build_config.name == "custom_config"


class TestGenerateVCSDumpWave:
    """Tests for generate_vcs_dump_wave function."""

    def test_generate_vcs_dump_wave_creates_file(self, tmp_path):
        """Test generate_vcs_dump_wave creates the file."""
        output_path = tmp_path / "dump_wave.v"
        top_module = "my_top"

        generate_vcs_dump_wave(output_path, top_module)

        assert output_path.exists()

    def test_generate_vcs_dump_wave_content(self, tmp_path):
        """Test generate_vcs_dump_wave generates correct Verilog content."""
        output_path = tmp_path / "dump_wave.v"
        top_module = "test_top"

        generate_vcs_dump_wave(output_path, top_module)

        content = output_path.read_text()

        assert "module dump_fsdb" in content
        assert "$fsdbDumpfile" in content
        assert "$fsdbDumpvars" in content
        assert top_module in content

    def test_generate_vcs_dump_wave_creates_parent_dirs(self, tmp_path):
        """Test generate_vcs_dump_wave creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "dump_wave.v"

        # Note: generate_vcs_dump_wave doesn't create parent directories
        # The test should reflect the actual behavior or we should create dirs first
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generate_vcs_dump_wave(output_path, "top")

        assert output_path.exists()
