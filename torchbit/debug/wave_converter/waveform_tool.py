"""
Waveform analysis tools.

Provides high-level functions for analyzing waveform files (FST/VCD format).
Assumes input waveform files are in correct format and uses assert for validation.
"""

import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal

from .wal_parser import WalParser, WalParserError


__all__ = [
    "posedge_dump_to_csv",
    "list_signals",
    "check_signals",
    "count_edges",
]


def _validate_wave_postfix(
    path: str,
    allowed: Tuple[str, ...] = (".fst", ".vcd")
) -> None:
    """
    Validate waveform file has correct postfix.

    Args:
        path: Path to waveform file.
        allowed: Tuple of allowed file extensions.

    Raises:
        AssertionError: If file extension is not in allowed list.
    """
    ext = Path(path).suffix.lower()
    allowed_str = ", ".join(allowed)
    assert ext in allowed, f"Unsupported waveform format: {ext}. Expected: {allowed_str}"


def _write_or_print(content: str, output_path: Optional[str]) -> None:
    """
    Write content to file or print to stdout.

    Args:
        content: Content to write/print.
        output_path: Path to output file. If None, print to stdout.
    """
    if output_path is None:
        print(content)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)


def _get_signal_width(parser: WalParser, signal: str) -> int:
    """
    Get the width of a signal.

    Args:
        parser: WalParser instance.
        signal: Signal name.

    Returns:
        Width of the signal in bits.
    """
    from wal.core import read_wal_sexpr

    # Use WAL to get signal width
    wal_query = f"(length {signal})"
    parsed_expr = read_wal_sexpr(wal_query)
    result = parser._evaluator.eval(parsed_expr)

    return result if isinstance(result, int) else len(result) if result else 1


def posedge_dump_to_csv(
    wave_path: str,
    clk: str,
    signal_list: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    format: Literal["csv", "csv_with_header"] = "csv_with_header",
    delimiter: str = ",",
) -> int:
    """
    Dump signals at clock posedge to CSV.

    Args:
        wave_path: Path to FST/VCD waveform file.
        clk: Clock signal name for posedge detection.
        signal_list: List of signals to dump. If None, dump all signals.
        output_path: Path to output CSV file. If None, print to stdout.
        time_start: Start time in simulation units.
        time_end: End time in simulation units.
        format: Output format.
        delimiter: CSV delimiter.

    Returns:
        Number of samples dumped.

    Raises:
        AssertionError: If wave_path has invalid postfix.
        WalParserError: If waveform parsing fails.
    """
    _validate_wave_postfix(wave_path)

    with WalParser(wave_path) as parser:
        # Set time window
        if time_start is not None or time_end is not None:
            parser.set_window(time_start, time_end)

        # Get signal list
        if signal_list is None:
            signal_list = parser.get_signal_names()

        # Sample at posedge
        result = parser.sample_posedge(clk, signal_list)

        # Build CSV content
        lines = []
        if format == "csv_with_header":
            header = ["time"] + signal_list
            lines.append(delimiter.join(str(h) for h in header))

        for i, timestamp in enumerate(result.timestamps):
            row = [str(timestamp)]
            row += [str(v) for v in result.signals[i]]
            lines.append(delimiter.join(row))

        csv_content = "\n".join(lines) + ("\n" if lines else "")

        _write_or_print(csv_content, output_path)

        return len(result.timestamps)


def list_signals(
    wave_path: str,
    output_path: Optional[str] = None,
    include_internal: bool = False,
    pattern: Optional[str] = None,
) -> List[str]:
    """
    List all signal names (full path) in waveform.

    Args:
        wave_path: Path to FST/VCD waveform file.
        output_path: Path to output file. If None, print to stdout.
        include_internal: If True, include internal signals ($*).
        pattern: Optional regex pattern to filter signals.

    Returns:
        List of signal names.

    Raises:
        AssertionError: If wave_path has invalid postfix.
        WalParserError: If waveform parsing fails.
    """
    _validate_wave_postfix(wave_path)

    with WalParser(wave_path) as parser:
        signals = parser.get_signal_names(include_internal=include_internal)

        # Apply regex pattern filter if provided
        if pattern is not None:
            regex = re.compile(pattern)
            signals = [s for s in signals if regex.search(s)]

        # Build output content
        content = "\n".join(signals) + ("\n" if signals else "")

        _write_or_print(content, output_path)

        return signals


def check_signals(
    wave_path: str,
    signal_list: List[str],
    timestamp: float,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check signal values at a specific timestamp.

    Args:
        wave_path: Path to FST/VCD waveform file.
        signal_list: List of signals to check.
        timestamp: Timestamp to check signal values at.
        output_path: Path to output file. If None, print to stdout.

    Returns:
        Dictionary mapping signal names to values.

    Raises:
        AssertionError: If wave_path has invalid postfix.
        WalParserError: If waveform parsing fails.
    """
    _validate_wave_postfix(wave_path)

    from wal.core import read_wal_sexpr

    with WalParser(wave_path) as parser:
        values = {}

        for signal in signal_list:
            # Build WAL query to get signal value at timestamp
            wal_query = f"(at {timestamp} {signal})"
            parsed_expr = read_wal_sexpr(wal_query)
            result = parser._evaluator.eval(parsed_expr)
            values[signal] = result

        # Build output content
        lines = [f"# Signal values at timestamp {timestamp}"]
        for signal, value in values.items():
            lines.append(f"{signal} = {value}")

        content = "\n".join(lines) + "\n"

        _write_or_print(content, output_path)

        return values


def count_edges(
    wave_path: str,
    signal: str,
    clk: str,
    time_range: Optional[Tuple[float, float]] = None,
    is_posedge: bool = True,
    output_path: Optional[str] = None,
) -> int:
    """
    Count valid cycles (when signal is high/1) at clock edges.

    Example: If signal pattern is "0110000111" at posedge timestamps,
             returns 5 (number of times signal == 1).

    Args:
        wave_path: Path to FST/VCD waveform file.
        signal: Target signal to count (must be 1-bit signal).
        clk: Clock signal name for edge detection.
        time_range: (start_time, end_time) tuple. If None, full range.
        is_posedge: True for posedge, False for negedge.
        output_path: Path to output file. If None, print to stdout.

    Returns:
        Count of cycles where signal is high (value == 1).

    Raises:
        AssertionError: If wave_path has invalid postfix or signal is not 1-bit.
        WalParserError: If waveform parsing fails.
    """
    _validate_wave_postfix(wave_path)

    from wal.core import read_wal_sexpr

    with WalParser(wave_path) as parser:
        # Validate signal is 1-bit
        width = _get_signal_width(parser, signal)
        assert width == 1, f"Signal '{signal}' has width {width}, expected 1-bit signal"

        # Set time window if specified
        if time_range is not None:
            parser.set_window(time_range[0], time_range[1])

        # Build WAL query for edge detection
        if is_posedge:
            edge_condition = f"(&& (= {clk} 1) (= (prev {clk}) 0))"
        else:
            edge_condition = f"(&& (= {clk} 0) (= (prev {clk}) 1))"

        # Query to get signal value at each edge
        wal_query = f"""
        (list
          (map
            (lambda (t)
              (at t {signal})
            )
            (find {edge_condition})
          )
        )
        """

        parsed_expr = read_wal_sexpr(wal_query)
        result = parser._evaluator.eval(parsed_expr)

        # Count occurrences where signal is high (1)
        count = sum(1 for v in result if v == 1)

        # Build output content
        edge_type = "posedge" if is_posedge else "negedge"
        lines = [
            f"# Count result for signal '{signal}' at {clk} {edge_type}",
            f"# Time range: {time_range if time_range else 'full waveform'}",
            f"Count: {count}"
        ]
        content = "\n".join(lines) + "\n"

        _write_or_print(content, output_path)

        return count
