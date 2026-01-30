#!/usr/bin/env python3
"""
Entry point for Tile Mapping example.

This example demonstrates:
- TwoPortBuffer for backdoor memory access
- TileMapping for address translation and data layout
- Tile transpose operation: (w, c) layout -> (c, w) layout
- Instruction-based configuration (w, c, base addresses at runtime)
- Non-overlapping source and destination regions

DUT:
- tile_mover.v - Tile transpose module with configurable dimensions
- TwoPortBuffer - Used directly as memory
"""
import sys
from pathlib import Path

# Add testbench root to path (parent directory of this file)
TB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TB_ROOT.parent))

from torchbit.runner import Runner, FileConfig, BuildConfig, DEFAULT_VERILATOR_BUILD_CONFIG


def main():
    """Run the Tile Mapping example."""
    print("=" * 60)
    print("TorchBit Tile Mapping Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - TwoPortBuffer for backdoor memory access")
    print("  - TileMapping for address translation")
    print("  - Tile transpose: (w, c) layout -> (c, w) layout")
    print("  - Instruction-based configuration (runtime)")
    print("  - Non-overlapping source/destination regions")
    print()
    print("Data Layout:")
    print("  - Source: (w, c) * (cs) where cs=32-bit data width")
    print("  - Destination: (c, w) * (cs) transposed layout")
    print()
    print("DUTs:")
    print("  - tile_mover.v - Configurable tile transpose module")
    print("  - TwoPortBuffer - Shared memory for source/destination")
    print()
    print("-" * 60)

    # Set up file configuration
    file_config = FileConfig(
        name="tile_mover",
        sources=[
            TB_ROOT / "src" / "rtl" / "tile_mover.v",
        ],
        top_design="tile_mover",
        includes=[]
    )

    # Set up build configuration
    build_config = DEFAULT_VERILATOR_BUILD_CONFIG

    # Create runner
    runner = Runner(file_config, build_config, current_dir=TB_ROOT)

    print()
    print("Building and running tests...")
    print()

    # Run the basic transpose test
    try:
        runner.test("tb.dut_tile_mover.tests.test_transpose_basic")
        print()
        print("=" * 60)
        print("Basic test completed successfully!")
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
