"""
Data Conversion Example - Entry Point

This script demonstrates Vector and Matrix classes for converting
between PyTorch tensors and hardware-compatible formats.

No DUT is required - this is a pure Python example.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the tests directly
from test_data_convert import main as run_tests


def main():
    """Run the data conversion example."""
    print("=" * 60)
    print("TorchBit Data Conversion Example")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  - Vector class: 1D tensor <-> integer conversion")
    print("  - Matrix class: 2D tensor <-> file conversion")
    print("  - Support for multiple dtypes")
    print("  - Roundtrip verification")
    print()
    print("No DUT required - pure Python example")
    print()
    print("-" * 60)
    print()

    return run_tests()


if __name__ == "__main__":
    sys.exit(main())
