"""
Waveform conversion utilities.

Converts VCD/FST/FSDB waveforms to CSV with posedge sampling.

Modules:
    - file_converter: FSDB to FST conversion and high-level workflow
    - wal_parser: WAL-based waveform parsing
    - waveform_tool: Waveform analysis tools
"""

from .file_converter import (
    FsdbConverter,
    FsdbConverterError,
    convert_fsdb_to_fst,
    convert_wave_to_text,
)

from .wal_parser import (
    WalParser,
    WalParserError,
    SampleResult,
    analyze_valid_data
)

from .waveform_tool import (
    posedge_dump_to_csv,
    list_signals,
    check_signals,
    count_edges,
)

__all__ = [
    # Main interface (from file_converter)
    "convert_wave_to_text",
    "convert_fsdb_to_fst",

    # Waveform analysis tools
    "posedge_dump_to_csv",
    "list_signals",
    "check_signals",
    "count_edges",

    # Classes (for advanced usage)
    "FsdbConverter",
    "WalParser",
    "SampleResult",

    # Functions
    "analyze_valid_data",

    # Exceptions
    "FsdbConverterError",
    "WalParserError",
]
