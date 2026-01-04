"""
Simulator runner utilities for Cocotb.

This module provides high-level interfaces for compiling and running
simulations with Verilator and VCS simulators.

Core concepts:
1. Runner: Cocotb simulator wrapper for Verilator and VCS
2. FileConfig: Configuration for HDL source files and top-level module
3. BuildConfig: Simulator build configuration
4. read_filelist: Parse HDL source filelists

Example:
    >>> from torchbit.runner import (
    ...     Runner, FileConfig, BuildConfig,
    ...     DEFAULT_VERILATOR_BUILD_CONFIG, read_filelist
    ... )
    >>>
    >>> # Read source files from filelist
    >>> files = read_filelist("design.f", base_path=".")
    >>>
    >>> # Configure source files
    >>> config = FileConfig(
    ...     name="my_design",
    ...     sources=files,
    ...     top_design="top"
    ... )
    >>>
    >>> # Create runner and run test
    >>> runner = Runner(config, DEFAULT_VERILATOR_BUILD_CONFIG)
    >>> runner.test("my_test")  # Runs my_test.py

Pre-configured build configurations:
- DEFAULT_VERILATOR_BUILD_CONFIG: Verilator with multi-threading and FST tracing
- DEFAULT_VCS_BUILD_CONFIG: VCS with timing checks disabled and FSDB tracing

Notes:
    - Verilator 5.036 has known issues with FST assertions
    - WSL Ubuntu 22.04 is the recommended environment for Verilator
    - CentOS 7 is recommended for VCS
"""
from .config import *
from .runner import *
from .filelist import *
