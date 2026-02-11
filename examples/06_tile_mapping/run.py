#!/usr/bin/env python3
"""
Tile Mapping Example - Entry Point

Demonstrates TileMapping and AddressMapping for tensor-to-memory layout
with spatial/temporal dimension mapping.

DUT: tile_mover.v - Tile transpose: (b, w, c) -> (w, b, c)
Requires cocotb + Verilator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchbit.runner import Runner, FileConfig, DEFAULT_VERILATOR_BUILD_CONFIG

EXAMPLE_DIR = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("TorchBit Tile Mapping Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - TileMapping for tensor-to-memory layout (einops)")
    print("  - AddressMapping for multi-dim -> flat address translation")
    print("  - Spatial mapping: cs=4 elements per cycle (parallel)")
    print("  - Tile transpose: (b, w, c) -> (w, b, c)")
    print()
    print("DUT: tile_mover.v with TwoPortBuffer as shared memory")
    print()
    print("-" * 60)

    file_config = FileConfig(
        name="tile_mover",
        sources=[EXAMPLE_DIR / "dut" / "tile_mover.v"],
        top_design="tile_mover",
        includes=[],
    )

    runner = Runner(file_config, DEFAULT_VERILATOR_BUILD_CONFIG, current_dir=EXAMPLE_DIR)

    print()
    print("Building and running tests...")
    print()

    try:
        runner.test("test_tile_mapping")
        print()
        print("=" * 60)
        print("Tests completed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print()
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
