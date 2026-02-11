"""Transfer strategy abstractions for verification components.

Provides pluggable strategies that control when data transfers occur,
enabling greedy, random-backpressure, burst, and throttled transfer
patterns for drivers and receivers.
"""
import random
from abc import ABC, abstractmethod


class TransferStrategy(ABC):
    """Abstract base for transfer timing strategies.

    Strategies control *when* a driver asserts valid or a receiver asserts
    ready.  They are pure-Python objects with no cocotb dependency, making
    them independently testable.

    Subclasses must implement ``should_transfer(cycle)`` which returns
    ``True`` when the component should participate in a handshake on the
    given cycle.
    """

    @abstractmethod
    def should_transfer(self, cycle: int) -> bool:
        """Decide whether to transfer on this cycle.

        Args:
            cycle: Zero-based cycle counter since ``run()`` started.

        Returns:
            True if the component should assert valid/ready this cycle.
        """
        ...

    def reset(self) -> None:
        """Reset internal state for reuse across multiple runs."""
        pass


class GreedyStrategy(TransferStrategy):
    """Transfer whenever the channel is ready (default behaviour)."""

    def should_transfer(self, cycle: int) -> bool:
        return True


class RandomBackpressure(TransferStrategy):
    """Randomly stall with configurable probability.

    Useful for stress-testing flow-control logic by injecting random
    back-pressure.

    Args:
        stall_prob: Probability of stalling on any given cycle (0.0–1.0).
        seed: Optional RNG seed for reproducible runs.

    Example:
        >>> strategy = RandomBackpressure(stall_prob=0.3, seed=42)
        >>> strategy.should_transfer(0)  # ~70% chance of True
    """

    def __init__(self, stall_prob: float = 0.3, seed: int | None = None):
        assert 0.0 <= stall_prob <= 1.0, "stall_prob must be in [0.0, 1.0]"
        self.stall_prob = stall_prob
        self.seed = seed
        self._rng = random.Random(seed)

    def should_transfer(self, cycle: int) -> bool:
        return self._rng.random() >= self.stall_prob

    def reset(self) -> None:
        self._rng = random.Random(self.seed)


class BurstStrategy(TransferStrategy):
    """Send *burst_len* items, pause *pause_cycles* cycles, repeat.

    Args:
        burst_len: Number of consecutive transfer cycles per burst.
        pause_cycles: Number of idle cycles between bursts.

    Example:
        >>> strategy = BurstStrategy(burst_len=4, pause_cycles=2)
        >>> [strategy.should_transfer(c) for c in range(8)]
        [True, True, True, True, False, False, True, True]
    """

    def __init__(self, burst_len: int = 8, pause_cycles: int = 4):
        assert burst_len > 0, "burst_len must be > 0"
        assert pause_cycles >= 0, "pause_cycles must be >= 0"
        self.burst_len = burst_len
        self.pause_cycles = pause_cycles

    def should_transfer(self, cycle: int) -> bool:
        period = self.burst_len + self.pause_cycles
        return (cycle % period) < self.burst_len


class ThrottledStrategy(TransferStrategy):
    """Transfer at most once every *min_interval* cycles.

    Args:
        min_interval: Minimum number of cycles between transfers.

    Example:
        >>> strategy = ThrottledStrategy(min_interval=3)
        >>> [strategy.should_transfer(c) for c in range(9)]
        [True, False, False, True, False, False, True, False, False]
    """

    def __init__(self, min_interval: int = 3):
        assert min_interval > 0, "min_interval must be > 0"
        self.min_interval = min_interval

    def should_transfer(self, cycle: int) -> bool:
        return (cycle % self.min_interval) == 0
