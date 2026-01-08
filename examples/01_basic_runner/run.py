"""
Basic Runner Example - Entry Point

This script demonstrates the minimal TorchBit Runner setup for
compiling and running a simple DUT with cocotb.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchbit.runner import (
    Runner,
    FileConfig,
    BuildConfig,
    DEFAULT_VERILATOR_BUILD_CONFIG,
)


def main():
    """Run the basic counter example."""
    print("=" * 60)
    print("TorchBit Basic Runner Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - Setting up FileConfig with Verilog sources")
    print("  - Using BuildConfig for Verilator")
    print("  - Running a minimal cocotb test")
    print("  - Generating waveform output (FST format)")
    print()
    print("DUT: simple_counter.v - Parameterized counter")
    print("Test: test_simple_counter.py - Basic compile/run test")
    print()
    print("Expected outputs:")
    print("  - Build directory: sim_basic_counter_default_verilator_test_simple_counter/")
    print("  - Waveform: waveform.fst")
    print()
    print("-" * 60)

    # Set up file configuration
    examples_dir = Path(__file__).parent
    dut_path = examples_dir / "dut"

    file_config = FileConfig(
        name="basic_counter",
        sources=[dut_path / "simple_counter.v"],
        top_design="simple_counter",
    )

    # Set up build configuration (Verilator)
    build_config = DEFAULT_VERILATOR_BUILD_CONFIG

    # Create and run the runner
    runner = Runner(file_config, build_config, current_dir=examples_dir)

    print()
    print("Building and running test...")
    print()

    try:
        runner.test("test_simple_counter")
        print()
        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        print()
        print(f"Waveform available in:")
        print(f"  {examples_dir / 'sim_basic_counter_default_verilator_test_simple_counter/waveform.fst'}")
        print()
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
