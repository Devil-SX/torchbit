"""
BitStruct Usage Example - Entry Point

This script demonstrates BitField and BitStruct classes for
bit-level data manipulation in hardware verification.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the tests directly
from test_bit_struct import main as run_tests


def main():
    """Run the BitStruct example."""
    print("=" * 60)
    print("TorchBit BitStruct Usage Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - BitField and BitStruct classes")
    print("  - Packed data structure manipulation")
    print("  - Hardware register modeling")
    print("  - Bit-level operations")
    print()
    print("No DUT required - pure Python example")
    print()
    print("-" * 60)
    print()

    return run_tests()


if __name__ == "__main__":
    sys.exit(main())
