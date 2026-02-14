#!/usr/bin/env python3
"""
Quick entry point for UVM Pipeline example.

This is a simplified entry point for running the example.
For full testframework functionality, use tb/dut_pipe/main.py.
"""
import sys
from pathlib import Path

TB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TB_ROOT.parent))

from torchbit.runner import Runner, FileConfig, DEFAULT_VERILATOR_BUILD_CONFIG


def main():
    """Run the UVM Pipeline example."""
    print("=" * 60)
    print("TorchBit UVM Pipeline Example")
    print("=" * 60)
    print()
    print("This example demonstrates UVM-style verification:")
    print("  - TorchbitBFM for signal abstraction (drive/sample)")
    print("  - TorchbitScoreboard for expected-vs-actual comparison")
    print("  - VectorItem as sequence items in the scoreboard")
    print("  - TransferStrategy for configurable driver timing")
    print("  - Golden model for reference predictions")
    print()
    print("DUT:")
    print("  - pipe.sv — parameterized pipeline (WIDTH=32, DELAY=4)")
    print()
    print("For full testframework:")
    print("  cd tb/dut_pipe && python main.py")
    print()
    print("-" * 60)

    file_config = FileConfig(
        name="pipe",
        sources=[
            TB_ROOT / "src" / "rtl" / "pipe.sv",
        ],
        top_design="Pipe",
        includes=[],
    )
    build_config = DEFAULT_VERILATOR_BUILD_CONFIG
    runner = Runner(file_config, build_config, current_dir=TB_ROOT)

    print()
    print("Building and running tests...")
    print()

    try:
        runner.test("tb.dut_pipe.tests.test_basic")
        print()
        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        print()
        return 0
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
