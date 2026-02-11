"""Tests for FIFODriver and FIFOReceiver (construction, pure Python)."""
import pytest
from torchbit.tools.fifo import FIFODriver, FIFOReceiver
from torchbit.tools.strategy import (
    GreedyStrategy,
    RandomBackpressure,
    BurstStrategy,
    ThrottledStrategy,
)
from torchbit.core.logic_sequence import LogicSequence


class TestFIFODriverInit:
    def test_default_strategy(self):
        d = FIFODriver()
        assert isinstance(d.strategy, GreedyStrategy)
        assert d.debug is False
        assert isinstance(d.queue, LogicSequence)
        assert len(d.queue) == 0

    def test_custom_strategy(self):
        s = RandomBackpressure(stall_prob=0.5, seed=1)
        d = FIFODriver(strategy=s, debug=True)
        assert d.strategy is s
        assert d.debug is True

    def test_load(self):
        d = FIFODriver()
        d.load([0xA, 0xB, 0xC])
        assert list(d.queue) == [0xA, 0xB, 0xC]
        assert isinstance(d.queue, LogicSequence)

    def test_load_clears_timestamps(self):
        d = FIFODriver()
        d.timestamps = [(100,)]
        d.load([1, 2])
        assert d.timestamps == []

    def test_dump_time_empty(self):
        d = FIFODriver()
        assert d.dump_time() == []


class TestFIFOReceiverInit:
    def test_default_strategy(self):
        r = FIFOReceiver()
        assert isinstance(r.strategy, GreedyStrategy)
        assert r.debug is False
        assert isinstance(r.data, LogicSequence)
        assert len(r.data) == 0

    def test_custom_strategy(self):
        s = BurstStrategy(burst_len=4, pause_cycles=2)
        r = FIFOReceiver(strategy=s, debug=True)
        assert r.strategy is s
        assert r.debug is True

    def test_dump_empty(self):
        r = FIFOReceiver()
        assert list(r.dump()) == []

    def test_dump_time_empty(self):
        r = FIFOReceiver()
        assert r.dump_time() == []


class TestFIFOImports:
    """Verify FIFO components are importable from tools."""

    def test_import_from_tools(self):
        from torchbit.tools import FIFODriver, FIFOReceiver
        assert FIFODriver is not None
        assert FIFOReceiver is not None

    def test_import_strategies_from_tools(self):
        from torchbit.tools import (
            GreedyStrategy,
            RandomBackpressure,
            BurstStrategy,
            ThrottledStrategy,
        )
        assert GreedyStrategy is not None
        assert RandomBackpressure is not None
        assert BurstStrategy is not None
        assert ThrottledStrategy is not None
