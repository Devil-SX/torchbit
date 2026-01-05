"""
WAL-based waveform parsing utilities.

Provides efficient posedge sampling of waveform files.
Uses the new wal API (0.8+) with direct trace methods for better abstraction.
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

    This implementation uses the new wal API (0.8+) with direct trace methods
    instead of complex WAL queries, providing cleaner abstractions.
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

        self._container = None
        self._evaluator = None
        self._trace = None
        self._wave_path = None
        self._window_start = None
        self._window_end = None

        self.load(wave_path)

    def load(self, wave_path: str) -> None:
        """Load waveform file."""
        from wal.core import TraceContainer
        from wal.eval import SEval

        self._container = TraceContainer()
        self._container.load(wave_path)

        self._evaluator = SEval(self._container)
        self._wave_path = wave_path

        # Get the trace object (VCD/FST trace)
        traces = list(self._container.traces.values())
        if not traces:
            raise WalParserError(f"No traces found in waveform file: {wave_path}")
        self._trace = traces[0]

    def set_window(self, time_start: Optional[float] = None,
                   time_end: Optional[float] = None) -> None:
        """
        Set time window for sampling.

        Args:
            time_start: Start time (None = from beginning).
            time_end: End time (None = to end).

        Note: In the new wal API, this is implemented as index filtering
        rather than trace modification.
        """
        self._window_start = time_start
        self._window_end = time_end

    def _filter_indices_by_window(self, indices: List[int]) -> List[int]:
        """Filter indices based on time window."""
        if self._window_start is None and self._window_end is None:
            return indices

        timestamps = self._trace.timestamps
        filtered = []
        for idx in indices:
            ts = timestamps[idx]
            if self._window_start is not None and ts < self._window_start:
                continue
            if self._window_end is not None and ts > self._window_end:
                continue
            filtered.append(idx)
        return filtered

    def get_signal_names(self, include_internal: bool = False) -> List[str]:
        """
        Get list of signal names in waveform.

        Args:
            include_internal: If False, filter out internal signals ($*).

        Returns:
            List of signal names.
        """
        names = self._container.signals
        if not include_internal:
            names = [n for n in names if not n.startswith("$")]
        return names

    def _find_posedge_indices(self, clk_name: str) -> List[int]:
        """
        Find indices where clock has rising edge.

        A rising edge is where clk=1 and previous clk=0.

        Args:
            clk_name: Clock signal name.

        Returns:
            List of indices where posedge occurs.
        """
        from wal.core import read_wal_sexpr

        # Find all indices where clk=1
        clk_high_query = f'(find (= {clk_name} 1))'
        parsed = read_wal_sexpr(clk_high_query)
        clk_high_indices = self._evaluator.eval(parsed)

        # Filter to only posedge (clk=1 and previous clk=0)
        posedge_indices = []
        for idx in clk_high_indices:
            if idx == 0:
                continue  # Can't be posedge at index 0
            prev_val = self._trace.signal_value(clk_name, idx - 1)
            if prev_val == 0:
                posedge_indices.append(idx)

        return self._filter_indices_by_window(posedge_indices)

    def _find_posedge_indices_with_condition(
        self, clk_name: str, condition: str
    ) -> List[int]:
        """
        Find indices where clock has rising edge AND condition is true.

        Args:
            clk_name: Clock signal name.
            condition: WAL condition expression (e.g., "(= top.valid 1)").

        Returns:
            List of indices where posedge with condition occurs.
        """
        from wal.core import read_wal_sexpr

        # Find all indices where clk=1
        clk_high_query = f'(find (= {clk_name} 1))'
        parsed = read_wal_sexpr(clk_high_query)
        clk_high_indices = self._evaluator.eval(parsed)

        # Find indices where condition is true
        cond_query = f'(find {condition})'
        parsed = read_wal_sexpr(cond_query)
        cond_indices = set(self._evaluator.eval(parsed))

        # Filter to posedge where condition is also true
        posedge_indices = []
        for idx in clk_high_indices:
            if idx == 0:
                continue
            if idx not in cond_indices:
                continue
            prev_val = self._trace.signal_value(clk_name, idx - 1)
            if prev_val == 0:
                posedge_indices.append(idx)

        return self._filter_indices_by_window(posedge_indices)

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

        # Remove clock from signal names to avoid duplicate
        signal_names = [s for s in signal_names if s != clk_name]

        # Find posedge indices
        posedge_indices = self._find_posedge_indices(clk_name)

        # Sample signals at those indices
        timestamps = []
        signal_values = []

        for idx in posedge_indices:
            timestamps.append(self._trace.timestamps[idx])
            row = [self._trace.signal_value(sig, idx) for sig in signal_names]
            signal_values.append(row)

        return SampleResult(timestamps=timestamps, signals=signal_values)

    def sample_posedge_with_condition(self, clk_name: str, condition: str,
                                       signal_names: Optional[List[str]] = None
                                       ) -> SampleResult:
        """
        Sample signals at clock posedge with additional condition.

        Args:
            clk_name: Clock signal name for posedge detection.
            condition: Additional condition string (e.g., "valid == 1").
                Should be a WAL expression like "(= top.valid 1)".
            signal_names: List of signals to sample. If None, samples all.

        Returns:
            SampleResult with timestamps and signal values.
        """
        if signal_names is None:
            signal_names = self.get_signal_names()

        # Remove clock from signal names to avoid duplicate
        signal_names = [s for s in signal_names if s != clk_name]

        # Find posedge indices with condition
        posedge_indices = self._find_posedge_indices_with_condition(
            clk_name, condition
        )

        # Sample signals at those indices
        timestamps = []
        signal_values = []

        for idx in posedge_indices:
            timestamps.append(self._trace.timestamps[idx])
            row = [self._trace.signal_value(sig, idx) for sig in signal_names]
            signal_values.append(row)

        return SampleResult(timestamps=timestamps, signals=signal_values)

    @property
    def time_range(self) -> Tuple[float, float]:
        """Get (start_time, end_time) of waveform."""
        timestamps = self._trace.timestamps
        if not timestamps:
            return (0, 0)
        return (timestamps[0], timestamps[-1])

    def close(self) -> None:
        """Release waveform resources."""
        self._container = None
        self._evaluator = None
        self._trace = None

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
        # Build condition for valid=1
        condition = f"(= {valid_name} 1)"

        result = parser.sample_posedge_with_condition(
            clk_name,
            condition,
            [data_name]
        )

        times = result.timestamps
        data_values = [sig[0] for sig in result.signals]

        return times, data_values
