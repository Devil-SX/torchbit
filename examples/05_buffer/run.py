#!/usr/bin/env python3
"""
Quick entry point for MemoryMoverBuffer example.

This is a simplified entry point for running the example.
For full testframework functionality, use tb/dut_buffer/main.py.
"""
import sys
from pathlib import Path

# Add testbench root to path (parent directory of this file)
TB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TB_ROOT.parent))

from torchbit.runner import Runner, FileConfig, BuildConfig, DEFAULT_VERILATOR_BUILD_CONFIG


def main():
    """Run the MemoryMover example."""
    print("=" * 60)
    print("TorchBit Buffer Usage Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - TwoPortBuffer for backdoor memory access")
    print("  - init_from_tensor() to load data into memory")
    print("  - dump_to_tensor() to read data from memory")
    print("  - TileMapping for address translation")
    print("  - Memory mover DUT that copies data between regions")
    print()
    print("DUTs:")
    print("  - memory_mover.v - Copies data between memory regions")
    print("  - TwoPortBuffer - Used directly as memory (no separate RAM needed)")
    print()
    print("For full testframework:")
    print("  cd tb/dut_buffer && python main.py")
    print()
    print("-" * 60)

    # Set up file configuration
    file_config = FileConfig(
        name="memory_mover",
        sources=[
            TB_ROOT / "src" / "rtl" / "memory_mover.v",
        ],
        top_design="memory_mover",
        includes=[]
    )

    # Set up build configuration
    build_config = DEFAULT_VERILATOR_BUILD_CONFIG

    # Create runner
    runner = Runner(file_config, build_config, current_dir=TB_ROOT)

    print()
    print("Building and running tests...")
    print()

    # Run the basic copy test
    try:
        runner.test("tb.dut_buffer.tests.test_copy_basic")
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
