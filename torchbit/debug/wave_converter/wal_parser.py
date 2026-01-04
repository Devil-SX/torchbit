"""
WAL-based waveform parsing utilities.

Provides efficient posedge sampling of waveform files.
"""

from typing import List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class SampleResult:
    """Result of waveform sampling."""
    timestamps: List[float]
    signals: List[List[Any]]


class WalParserError(Exception):
    """Error during WAL parsing."""
    pass


class WalParser:
    """
    Waveform parser using WAL (Waveform Analysis Language).

    Supports FST and VCD formats directly.
    """

    def __init__(self, wave_path: str):
        """
        Initialize parser with waveform file.

        Args:
            wave_path: Path to FST or VCD file.

        Raises:
            WalParserError: If file cannot be loaded.
        """
        from wal.core import TraceContainer
        from wal.eval import SEval
        from wal.core import read_wal_sexpr

        self._container = None
        self._evaluator = None
        self._wal_parse = staticmethod(read_wal_sexpr)

        self.load(wave_path)

    def load(self, wave_path: str) -> None:
        """Load waveform file."""
        from wal.core import TraceContainer
        from wal.eval import SEval

        self._container = TraceContainer()
        self._container.load(wave_path)

        self._evaluator = SEval(self._container)
        self._wave_path = wave_path

    def set_window(self, time_start: Optional[float] = None,
                   time_end: Optional[float] = None) -> None:
        """
        Set time window for sampling.

        Args:
            time_start: Start time (None = from beginning).
            time_end: End time (None = to end).
        """
        self._container.set_window(time_start, time_end)

    def get_signal_names(self, include_internal: bool = False) -> List[str]:
        """
        Get list of signal names in waveform.

        Args:
            include_internal: If False, filter out internal signals ($*).

        Returns:
            List of signal names.
        """
        names = self._container.get_all_signal_names()
        if not include_internal:
            names = [n for n in names if not n.startswith("$")]
        return names

    def sample_posedge(self, clk_name: str,
                       signal_names: Optional[List[str]] = None) -> SampleResult:
        """
        Sample signals at clock posedge.

        Args:
            clk_name: Clock signal name for posedge detection.
            signal_names: List of signals to sample. If None, samples all.

        Returns:
            SampleResult with timestamps and signal values.
        """
        if signal_names is None:
            signal_names = self.get_signal_names()

        # Build WAL query
        wal_query = self._build_posedge_query(clk_name, signal_names)

        # Execute query
        parsed_expr = self._wal_parse(wal_query)
        result = self._evaluator.eval(parsed_expr)

        # Parse result: list of [time, sig1, sig2, ...]
        timestamps = [row[0] for row in result]
        signal_values = [row[1:] for row in result]

        return SampleResult(timestamps=timestamps, signals=signal_values)

    def sample_posedge_with_condition(self, clk_name: str, condition: str,
                                       signal_names: Optional[List[str]] = None
                                       ) -> SampleResult:
        """
        Sample signals at clock posedge with additional condition.

        Args:
            clk_name: Clock signal name for posedge detection.
            condition: Additional condition string (e.g., "valid == 1").
            signal_names: List of signals to sample. If None, samples all.

        Returns:
            SampleResult with timestamps and signal values.
        """
        if signal_names is None:
            signal_names = self.get_signal_names()

        # Build WAL query with condition
        wal_query = self._build_posedge_query_with_condition(
            clk_name, condition, signal_names
        )

        # Execute query
        parsed_expr = self._wal_parse(wal_query)
        result = self._evaluator.eval(parsed_expr)

        # Parse result: list of [time, sig1, sig2, ...]
        timestamps = [row[0] for row in result]
        signal_values = [row[1:] for row in result]

        return SampleResult(timestamps=timestamps, signals=signal_values)

    def _build_posedge_query(self, clk_name: str,
                             signal_names: List[str]) -> str:
        """
        Build WAL query for posedge sampling.

        Query pattern:
        (find (&& (= clk 1) (= (prev clk) 0)))  - find posedge
        (at t signal_name)                      - sample signal at time t
        """
        at_calls = "\n          ".join(
            f"(at t {name})" for name in signal_names
        )

        wal_query = f"""
        (list
          (map
            (lambda (t)
              (list
                t
                {at_calls}
              )
            )
            (find (&& (= {clk_name} 1) (= (prev {clk_name}) 0)))
          )
        )
        """
        return wal_query

    def _build_posedge_query_with_condition(self, clk_name: str,
                                            condition: str,
                                            signal_names: List[str]) -> str:
        """
        Build WAL query for posedge sampling with additional condition.

        Args:
            clk_name: Clock signal name.
            condition: Additional condition string.
            signal_names: List of signals to sample.
        """
        at_calls = "\n          ".join(
            f"(at t {name})" for name in signal_names
        )

        wal_query = f"""
        (list
          (map
            (lambda (t)
              (list
                t
                {at_calls}
              )
            )
            (find (&& (= {clk_name} 1) (= (prev {clk_name}) 0) {condition}))
          )
        )
        """
        return wal_query

    @property
    def time_range(self) -> Tuple[float, float]:
        """Get (start_time, end_time) of waveform."""
        return self._container.time_range()

    def close(self) -> None:
        """Release waveform resources."""
        self._container = None
        self._evaluator = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return f"WalParser(path={self._wave_path!r})"


def analyze_valid_data(waveform_path: str, clk_name: str, valid_name: str,
                       data_name: str) -> Tuple[List, List]:
    """
    Analyze valid-data pairs from a waveform using WAL queries.

    Uses the WAL (Waveform Analysis Language) to extract timing relationships
    between clock, valid, and data signals. Finds all rising clock edges where
    valid is asserted and returns the timestamps and corresponding data values.

    Args:
        waveform_path: Path to the waveform file (FST/VCD format).
        clk_name: Name of the clock signal in the waveform hierarchy
            (e.g., "top.dut.clk").
        valid_name: Name of the valid signal (e.g., "top.dut.valid").
        data_name: Name of the data signal (e.g., "top.dut.data_out").

    Returns:
        Tuple of (timestamps, data_values):
        - timestamps: List of simulation times when valid data was captured
        - data_values: List of data values at those timestamps

    Example:
        >>> times, data = analyze_valid_data(
        ...     "dump.fst",
        ...     clk_name="top.dut.clk",
        ...     valid_name="top.dut.valid",
        ...     data_name="top.dut.data_out"
        ... )
        >>> print(f"Found {len(times)} valid data pairs")
        >>> for t, d in zip(times[:5], data[:5]):
        ...     print(f"  t={t}, data={hex(d)}")
    """
    with WalParser(waveform_path) as parser:
        result = parser.sample_posedge_with_condition(
            clk_name,
            f"(= {valid_name} 1)",
            [data_name]
        )

        times = result.timestamps
        data_values = [sig[0] for sig in result.signals]

        return times, data_values
