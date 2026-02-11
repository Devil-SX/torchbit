#!/usr/bin/env python3
"""
UVM Basics Example - Entry Point

Demonstrates basic UVM components from torchbit.uvm.
Pure Python - no cocotb or simulator needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_uvm_basics import main

if __name__ == "__main__":
    sys.exit(main())
