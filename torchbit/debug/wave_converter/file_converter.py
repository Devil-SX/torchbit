"""
External file format conversion utilities.

Handles conversion of proprietary formats to FST for WAL processing.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

from .wal_parser import WalParser
from .waveform_tool import posedge_dump_to_csv


@dataclass
class FsdbConverterError(Exception):
    """Error during FSDB conversion."""
    message: str
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


class FsdbConverter:
    """
    Converter for FSDB to FST format.

    Uses fsdb2fst utility from VCS/Verdi installation.
    """

    def __init__(self, fsdb2fst_path: Optional[str] = None):
        """
        Initialize converter.

        Args:
            fsdb2fst_path: Path to fsdb2fst binary. If None, searches PATH.
        """
        self.fsdb2fst_path = fsdb2fst_path or shutil.which("fsdb2fst")
        if self.fsdb2fst_path is None:
            raise FsdbConverterError(
                "fsdb2fst not found. Please ensure VCS/Verdi is installed "
                "and fsdb2fst is in PATH."
            )

    def convert(self, fsdb_path: str, fst_path: str,
                reuse_existing: bool = True) -> str:
        """
        Convert FSDB file to FST format.

        Args:
            fsdb_path: Path to input FSDB file.
            fst_path: Path to output FST file.
            reuse_existing: If True and fst_path exists, skip conversion.

        Returns:
            Path to the FST file.

        Raises:
            FsdbConverterError: If conversion fails.
            FileNotFoundError: If input file doesn't exist.
        """
        fsdb_path = Path(fsdb_path)
        fst_path = Path(fst_path)

        if not fsdb_path.exists():
            raise FileNotFoundError(f"FSDB file not found: {fsdb_path}")

        # Check if we can reuse existing FST
        if reuse_existing and fst_path.exists():
            return str(fst_path)

        # Ensure output directory exists
        fst_path.parent.mkdir(parents=True, exist_ok=True)

        # Run conversion
        cmd = [self.fsdb2fst_path, str(fsdb_path), str(fst_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise FsdbConverterError(
                f"fsdb2fst failed with code {result.returncode}",
                return_code=result.returncode,
                stderr=result.stderr
            )

        return str(fst_path)

    def __repr__(self) -> str:
        return f"FsdbConverter(fsdb2fst={self.fsdb2fst_path!r})"


def convert_fsdb_to_fst(
    fsdb_path: str,
    reuse: bool = True
) -> str:
    """
    Convert FSDB format file to FST format.

    This is a file format conversion only - it does not read signal data.
    The converted FST file can be used with waveform tools or for signal extraction.

    Args:
        fsdb_path: Path to FSDB file.
        reuse: If True, reuse existing FST file if present.

    Returns:
        Path to the converted FST file.

    Raises:
        FileNotFoundError: Input FSDB file not found.
        FsdbConverterError: FSDB to FST conversion failed.
    """
    fsdb_path_obj = Path(fsdb_path)
    if not fsdb_path_obj.exists():
        raise FileNotFoundError(f"FSDB file not found: {fsdb_path}")

    fst_path = fsdb_path.replace(".fsdb", ".fst")

    if reuse and Path(fst_path).exists():
        return fst_path

    converter = FsdbConverter()
    return converter.convert(fsdb_path, fst_path, reuse_existing=reuse)


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

    High-level convenience function that handles both format conversion
    (FSDB to FST) and signal extraction.

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
            fst_path = convert_fsdb_to_fst(
                str(wavefile_path),
                reuse=reuse_fst
            )
            wave_path = fst_path
        else:
            wave_path = str(wavefile_path)

        # Read signals and write CSV (using waveform_tool function)
        samples = posedge_dump_to_csv(
            wave_path, clk, signal_list=None,
            output_path=output_path,
            time_start=time_start, time_end=time_end,
            format=format, delimiter=delimiter
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
