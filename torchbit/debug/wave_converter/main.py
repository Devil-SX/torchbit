"""
Waveform to text conversion interface.

Converts waveform files (VCD/FST/FSDB) to CSV with posedge sampling.
"""

from pathlib import Path
from typing import Optional, Literal

from .file_converter import FsdbConverter, FsdbConverterError
from .wal_parser import WalParser, SampleResult


__all__ = ["convert_wave_to_text"]


def convert_wave_to_text(
    wavefile_path: str,
    output_path: str,
    clk: str,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    format: Literal["csv", "csv_with_header"] = "csv_with_header",
    delimiter: str = ",",
    reuse_fst: bool = True
) -> int:
    """
    Convert waveform to text format with posedge sampling.

    Flow:
        FSDB -> FST (via fsdb2fst) -> CSV (via WAL)
        VCD -> CSV (via WAL)
        FST -> CSV (via WAL)

    Args:
        wavefile_path: Path to waveform file (.vcd/.fst/.fsdb).
        output_path: Path to output CSV file.
        clk: Clock signal name for posedge detection (e.g., "top.dut.clk").
        time_start: Start time in simulation units (None = from beginning).
        time_end: End time in simulation units (None = to end).
        format: Output format - "csv" or "csv_with_header".
        delimiter: CSV delimiter (default: ",").
        reuse_fst: For FSDB input, reuse existing FST if present.

    Returns:
        Number of samples extracted.

    Raises:
        FileNotFoundError: Input file not found.
        ValueError: Unsupported format or invalid arguments.
        FsdbConverterError: FSDB to FST conversion failed.
        WalParserError: WAL parsing failed.
    """
    # Validate inputs
    wavefile_path = Path(wavefile_path)
    if not wavefile_path.exists():
        raise FileNotFoundError(f"Waveform file not found: {wavefile_path}")

    # Detect format
    fmt = _detect_format(wavefile_path)
    if fmt == "unknown":
        raise ValueError(f"Unsupported waveform format: {wavefile_path}")

    # Convert FSDB to FST if needed
    fst_path = None
    try:
        if fmt == "fsdb":
            fst_path = _convert_fsdb_to_fst(
                str(wavefile_path),
                reuse=reuse_fst
            )
            wave_path = fst_path
        else:
            wave_path = str(wavefile_path)

        # Parse and sample with WAL
        samples = _sample_and_write(
            wave_path, output_path, clk,
            time_start, time_end,
            format, delimiter
        )

        return samples

    finally:
        # Cleanup temporary FST file
        if fst_path and fst_path != str(wavefile_path):
            try:
                Path(fst_path).unlink(missing_ok=True)
            except OSError:
                pass  # Ignore cleanup errors


def _detect_format(path: Path) -> str:
    """Detect waveform format from file extension."""
    ext = path.suffix.lower()
    format_map = {
        ".fst": "fst",
        ".fsdb": "fsdb",
        ".vcd": "vcd",
    }
    return format_map.get(ext, "unknown")


def _convert_fsdb_to_fst(fsdb_path: str, reuse: bool = True) -> str:
    """Convert FSDB to FST using fsdb2fst."""
    fst_path = fsdb_path.replace(".fsdb", ".fst")

    if reuse and Path(fst_path).exists():
        return fst_path

    converter = FsdbConverter()
    return converter.convert(fsdb_path, fst_path, reuse_existing=reuse)


def _sample_and_write(
    wave_path: str,
    output_path: str,
    clk: str,
    time_start: Optional[float],
    time_end: Optional[float],
    format: str,
    delimiter: str
) -> int:
    """Load waveform, sample at posedge, write CSV."""
    with WalParser(wave_path) as parser:
        # Set time window
        if time_start is not None or time_end is not None:
            parser.set_window(time_start, time_end)

        # Get signal names
        signal_names = parser.get_signal_names()

        # Sample at posedge (clk is removed from sampled signals internally)
        result = parser.sample_posedge(clk, signal_names)

        # Remove clk from signal_names for header to match sampled data
        sampled_signal_names = [s for s in signal_names if s != clk]

        # Write CSV
        _write_csv(
            output_path, result,
            sampled_signal_names, format, delimiter
        )

        return len(result.timestamps)


def _write_csv(
    output_path: str,
    result: SampleResult,
    signal_names: list,
    format: str,
    delimiter: str
) -> None:
    """Write sampling result to CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header for csv_with_header
        if format == "csv_with_header":
            header = ["time"] + signal_names
            f.write(delimiter.join(str(h) for h in header) + "\n")

        # Write data rows
        for i, timestamp in enumerate(result.timestamps):
            row = [str(timestamp)]
            row += [str(v) for v in result.signals[i]]
            f.write(delimiter.join(row) + "\n")
