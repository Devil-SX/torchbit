"""Tests for TransferStrategy implementations (pure Python, no cocotb)."""
import pytest
from torchbit.tools.strategy import (
    GreedyStrategy,
    RandomBackpressure,
    BurstStrategy,
    ThrottledStrategy,
)


class TestGreedyStrategy:
    def test_always_true(self):
        s = GreedyStrategy()
        assert all(s.should_transfer(c) for c in range(100))


class TestRandomBackpressure:
    def test_deterministic_with_seed(self):
        s1 = RandomBackpressure(stall_prob=0.5, seed=42)
        s2 = RandomBackpressure(stall_prob=0.5, seed=42)
        r1 = [s1.should_transfer(c) for c in range(50)]
        r2 = [s2.should_transfer(c) for c in range(50)]
        assert r1 == r2

    def test_zero_stall_always_transfers(self):
        s = RandomBackpressure(stall_prob=0.0)
        assert all(s.should_transfer(c) for c in range(100))

    def test_full_stall_never_transfers(self):
        s = RandomBackpressure(stall_prob=1.0)
        assert not any(s.should_transfer(c) for c in range(100))

    def test_reset_reproduces(self):
        s = RandomBackpressure(stall_prob=0.5, seed=7)
        r1 = [s.should_transfer(c) for c in range(20)]
        s.reset()
        r2 = [s.should_transfer(c) for c in range(20)]
        assert r1 == r2

    def test_invalid_stall_prob(self):
        with pytest.raises(AssertionError):
            RandomBackpressure(stall_prob=-0.1)
        with pytest.raises(AssertionError):
            RandomBackpressure(stall_prob=1.1)

    def test_approximate_ratio(self):
        s = RandomBackpressure(stall_prob=0.3, seed=123)
        results = [s.should_transfer(c) for c in range(10000)]
        ratio = sum(results) / len(results)
        assert 0.65 < ratio < 0.75  # expect ~0.70


class TestBurstStrategy:
    def test_basic_pattern(self):
        s = BurstStrategy(burst_len=4, pause_cycles=2)
        pattern = [s.should_transfer(c) for c in range(12)]
        assert pattern == [
            True, True, True, True, False, False,
            True, True, True, True, False, False,
        ]

    def test_no_pause(self):
        s = BurstStrategy(burst_len=3, pause_cycles=0)
        assert all(s.should_transfer(c) for c in range(100))

    def test_single_burst(self):
        s = BurstStrategy(burst_len=1, pause_cycles=1)
        pattern = [s.should_transfer(c) for c in range(6)]
        assert pattern == [True, False, True, False, True, False]

    def test_invalid_burst_len(self):
        with pytest.raises(AssertionError):
            BurstStrategy(burst_len=0)

    def test_invalid_pause(self):
        with pytest.raises(AssertionError):
            BurstStrategy(pause_cycles=-1)


class TestThrottledStrategy:
    def test_basic_pattern(self):
        s = ThrottledStrategy(min_interval=3)
        pattern = [s.should_transfer(c) for c in range(9)]
        assert pattern == [True, False, False, True, False, False, True, False, False]

    def test_interval_1_is_greedy(self):
        s = ThrottledStrategy(min_interval=1)
        assert all(s.should_transfer(c) for c in range(100))

    def test_invalid_interval(self):
        with pytest.raises(AssertionError):
            ThrottledStrategy(min_interval=0)
