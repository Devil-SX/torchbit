"""
Waveform conversion utilities.

Converts VCD/FST/FSDB waveforms to CSV with posedge sampling.

Modules:
    - file_converter: FSDB to FST conversion
    - wal_parser: WAL-based waveform parsing
    - main: High-level conversion interface
"""

from .file_converter import (
    FsdbConverter,
    FsdbConverterError
)

from .wal_parser import (
    WalParser,
    WalParserError,
    SampleResult,
    analyze_valid_data
)

from .main import (
    convert_wave_to_text
)

__all__ = [
    # Main interface
    "convert_wave_to_text",

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
