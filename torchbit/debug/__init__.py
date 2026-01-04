"""
Debugging and verification utilities for hardware verification.

This module provides tools for:
1. compare: Tensor comparison with detailed metrics and visualization
2. Signal/SignalGroup: Waveform viewer integration (GTKWave)
3. compare_bit_struct: BitStruct field-by-field comparison
4. Temporal event visualization: Draw and analyze event timing
5. Waveform conversion: Convert VCD/FST/FSDB to CSV with posedge sampling

Typical workflow:
    >>> from torchbit.debug import compare
    >>> # Compare testbench output with golden reference
    >>> is_equal = compare(golden, result, rtol=1e-3, atol=1e-1)
    >>> # Output: Shape, max abs diff, max rel diff, MSE, pass/fail
"""

from .judge import compare
from .signallist import Signal, SignalGroup, generate_gtkwave_tcl
from .bit_struct_comparator import compare_bit_struct

# Waveform conversion utilities
from .wave_converter import (
    convert_wave_to_text,
    FsdbConverter,
    WalParser,
    SampleResult,
    analyze_valid_data,
    FsdbConverterError,
    WalParserError,
)

__all__ = [
    # Tensor comparison
    "compare",

    # Waveform viewer integration
    "Signal",
    "SignalGroup",
    "generate_gtkwave_tcl",

    # BitStruct comparison
    "compare_bit_struct",

    # Waveform conversion
    "convert_wave_to_text",
    "FsdbConverter",
    "WalParser",
    "SampleResult",
    "analyze_valid_data",
    "FsdbConverterError",
    "WalParserError",
]
