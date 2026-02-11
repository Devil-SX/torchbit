"""pyuvm-compatible scoreboard for torchbit data comparison."""
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TorchbitScoreboard:
    """Scoreboard that compares expected vs actual torchbit items.

    Works as standalone comparison engine without pyuvm.
    Use create_uvm_scoreboard() for a pyuvm-compatible version.
    """

    def __init__(self, name: str = "scoreboard"):
        self.name = name
        self.expected_queue = deque()
        self.actual_queue = deque()
        self.match_count = 0
        self.mismatch_count = 0
        self.errors = []

    def add_expected(self, item):
        """Add an expected item."""
        self.expected_queue.append(item)
        self._try_compare()

    def add_actual(self, item):
        """Add an actual item from DUT."""
        self.actual_queue.append(item)
        self._try_compare()

    def _try_compare(self):
        """Compare when both queues have items."""
        while self.expected_queue and self.actual_queue:
            exp = self.expected_queue.popleft()
            act = self.actual_queue.popleft()
            if exp == act:
                self.match_count += 1
                logger.debug(f"[{self.name}] Match #{self.match_count}: {act}")
            else:
                self.mismatch_count += 1
                msg = f"[{self.name}] MISMATCH #{self.mismatch_count}: expected={exp}, actual={act}"
                self.errors.append(msg)
                logger.error(msg)

    @property
    def passed(self) -> bool:
        """True if no mismatches and all items consumed."""
        return (
            self.mismatch_count == 0
            and len(self.expected_queue) == 0
            and len(self.actual_queue) == 0
        )

    def report(self) -> str:
        """Generate a summary report."""
        lines = [
            f"Scoreboard '{self.name}' Report:",
            f"  Matches: {self.match_count}",
            f"  Mismatches: {self.mismatch_count}",
            f"  Pending expected: {len(self.expected_queue)}",
            f"  Pending actual: {len(self.actual_queue)}",
            f"  Status: {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    {e}")
        return "\n".join(lines)


def create_uvm_scoreboard(name: str = "TorchbitUvmScoreboard"):
    """Factory that creates a uvm_component subclass with analysis exports.

    Requires pyuvm to be installed.
    """
    try:
        from pyuvm import uvm_component, uvm_tlm_analysis_fifo
    except ImportError:
        raise ImportError(
            "torchbit.uvm requires pyuvm. Install with: pip install pyuvm>=4.0.0"
        )

    class _TorchbitUvmScoreboard(uvm_component):
        def build_phase(self):
            super().build_phase()
            self.expected_fifo = uvm_tlm_analysis_fifo("expected_fifo", self)
            self.actual_fifo = uvm_tlm_analysis_fifo("actual_fifo", self)
            self.scoreboard = TorchbitScoreboard(self.get_name())

        def check_phase(self):
            while self.expected_fifo.can_get():
                _, item = self.expected_fifo.try_get()
                self.scoreboard.add_expected(item)
            while self.actual_fifo.can_get():
                _, item = self.actual_fifo.try_get()
                self.scoreboard.add_actual(item)

            assert self.scoreboard.passed, self.scoreboard.report()

    _TorchbitUvmScoreboard.__name__ = name
    _TorchbitUvmScoreboard.__qualname__ = name
    return _TorchbitUvmScoreboard
