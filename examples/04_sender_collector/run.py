"""
Sender/Collector Example - Entry Point

This script demonstrates the Sender and PoolCollector classes
for driving and capturing data from a hardware DUT.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchbit.runner import Runner, FileConfig, BuildConfig, DEFAULT_VERILATOR_BUILD_CONFIG


def main():
    """Run the Sender/Collector example."""
    print("=" * 60)
    print("TorchBit Sender/Collector Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - Sender class for driving stimulus to DUT")
    print("  - PoolCollector class for capturing DUT output")
    print("  - Flow control and timing visualization")
    print("  - Data verification and temporal event graphing")
    print()
    print("DUT: pipe.sv - Parameterized pipeline with delay")
    print("Test: test_pipe.py - Sender/Collector demonstration")
    print()
    print("Expected outputs:")
    print("  - Build directory: sim_pipe_default_verilator_test_pipe/")
    print("  - Timing graph: pipe_timing.png")
    print("  - Waveform: waveform.fst")
    print()
    print("-" * 60)

    # Import and run the test
    import test_pipe

    return 0


if __name__ == "__main__":
    sys.exit(main())
