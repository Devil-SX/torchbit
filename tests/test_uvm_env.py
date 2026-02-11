"""Tests for torchbit.uvm environment components (no pyuvm needed)."""
import pytest
from torchbit.uvm.scoreboard import TorchbitScoreboard
from torchbit.uvm.env import TorchbitEnv
from torchbit.uvm.test import TorchbitTest
from torchbit.uvm.seq_items import VectorItem, LogicSequenceItem
from torchbit.core.vector import Vector
from torchbit.core.logic_sequence import LogicSequence
import torch


class TestScoreboard:
    def test_matching_items(self):
        sb = TorchbitScoreboard("test_sb")
        v1 = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32))
        item1 = VectorItem("a", v1)
        item2 = VectorItem("b", v1)  # same data
        sb.add_expected(item1)
        sb.add_actual(item2)
        assert sb.passed
        assert sb.match_count == 1
        assert sb.mismatch_count == 0

    def test_mismatching_items(self):
        sb = TorchbitScoreboard("test_sb")
        v1 = Vector.from_array(torch.tensor([1.0], dtype=torch.float32))
        v2 = Vector.from_array(torch.tensor([2.0], dtype=torch.float32))
        sb.add_expected(VectorItem("exp", v1))
        sb.add_actual(VectorItem("act", v2))
        assert not sb.passed
        assert sb.mismatch_count == 1

    def test_logic_sequence_items(self):
        sb = TorchbitScoreboard("test_sb")
        seq1 = LogicSequence([0xDEAD, 0xBEEF])
        seq2 = LogicSequence([0xDEAD, 0xBEEF])
        sb.add_expected(LogicSequenceItem("exp", seq1))
        sb.add_actual(LogicSequenceItem("act", seq2))
        assert sb.passed

    def test_multiple_items(self):
        sb = TorchbitScoreboard()
        for i in range(10):
            v = Vector.from_array(torch.tensor([float(i)], dtype=torch.float32))
            sb.add_expected(VectorItem(f"exp_{i}", v))
            sb.add_actual(VectorItem(f"act_{i}", v))
        assert sb.passed
        assert sb.match_count == 10

    def test_report(self):
        sb = TorchbitScoreboard("report_test")
        report = sb.report()
        assert "PASS" in report
        assert "report_test" in report

    def test_pending_items_fail(self):
        sb = TorchbitScoreboard()
        sb.add_expected(VectorItem("exp", None))
        assert not sb.passed  # pending expected

    def test_empty_scoreboard_passes(self):
        sb = TorchbitScoreboard()
        assert sb.passed


class TestEnv:
    def test_create(self):
        env = TorchbitEnv("my_env")
        assert env.name == "my_env"
        assert env.agents == {}
        assert env.scoreboard is None

    def test_add_agent(self):
        env = TorchbitEnv()
        env.add_agent("input", "mock_agent")
        assert "input" in env.agents

    def test_set_scoreboard(self):
        env = TorchbitEnv()
        sb = TorchbitScoreboard()
        env.set_scoreboard(sb)
        assert env.scoreboard is sb


class TestUvmTest:
    def test_create(self):
        t = TorchbitTest("my_test")
        assert t.name == "my_test"
        assert t.env is None
