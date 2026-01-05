"""Verify which torchbit version is being imported.

Run this script to check if tests will use the development version
or the installed version from the conda environment.
"""

import torchbit
import sys
import os


def main():
    """Print torchbit import location and version info."""
    torchbit_path = torchbit.__file__
    torchbit_dir = os.path.dirname(os.path.abspath(torchbit_path))

    print("=" * 60)
    print("torchbit Import Verification")
    print("=" * 60)
    print(f"Location: {torchbit_path}")
    print(f"Version:  {getattr(torchbit, '__version__', 'unknown')}")

    # Detect which version is being used
    if "torchbit-test" in torchbit_path:
        print("\n[OK] Using DEVELOPMENT version")
        print("      Tests will run against the local source code.")
    elif "site-packages" in torchbit_path:
        print("\n[WARNING] Using INSTALLED version from conda env")
        print("          To use development version, run:")
        print("          export PYTHONPATH=/home/sdu/torchbit_project/torchbit-test:$PYTHONPATH")
    else:
        print(f"\n[UNKNOWN] torchbit from unexpected location")

    print("=" * 60)


if __name__ == "__main__":
    main()
