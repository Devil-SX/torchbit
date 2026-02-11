"""Tests for torchbit.uvm package (without pyuvm installed)."""
import pytest
import torch


class TestBFM:
    """Test BFM without cocotb (pure Python construction)."""

    def test_bfm_creation(self):
        from torchbit.uvm.bfm import TorchbitBFM

        class MockSignal:
            def __init__(self):
                self.value = 0

        dut = type("DUT", (), {"clk": MockSignal()})()
        bfm = TorchbitBFM(dut, dut.clk)
        assert bfm.dut is dut

    def test_bfm_add_signals(self):
        from torchbit.uvm.bfm import TorchbitBFM

        class MockSignal:
            def __init__(self):
                self.value = 0

        dut = type(
            "DUT",
            (),
            {"clk": MockSignal(), "din": MockSignal(), "dout": MockSignal()},
        )()
        bfm = TorchbitBFM(dut, dut.clk)
        bfm.add_input("data", dut.din)
        bfm.add_output("result", dut.dout)
        assert bfm.has_input("data")
        assert bfm.has_output("result")
        assert not bfm.has_input("result")

    def test_bfm_drive_sample(self):
        from torchbit.uvm.bfm import TorchbitBFM

        class MockSignal:
            def __init__(self):
                self.value = 0

        dut = type(
            "DUT",
            (),
            {"clk": MockSignal(), "din": MockSignal(), "dout": MockSignal()},
        )()
        bfm = TorchbitBFM(dut, dut.clk)
        bfm.add_input("data", dut.din)
        bfm.add_output("result", dut.dout)

        bfm.drive("data", 42)
        assert dut.din.value == 42

        dut.dout.value = 99
        assert bfm.sample("result") == 99


class TestSeqItems:
    def test_vector_item_creation(self):
        from torchbit.uvm.seq_items import VectorItem
        from torchbit.core.vector import Vector

        vec = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32))
        item = VectorItem("test", vec)
        assert item.name == "test"
        assert item.vector is vec

    def test_vector_item_equality(self):
        from torchbit.uvm.seq_items import VectorItem
        from torchbit.core.vector import Vector

        vec1 = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32))
        vec2 = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32))
        vec3 = Vector.from_array(torch.tensor([3.0, 4.0], dtype=torch.float32))

        assert VectorItem("a", vec1) == VectorItem("b", vec2)
        assert VectorItem("a", vec1) != VectorItem("b", vec3)

    def test_vector_item_repr(self):
        from torchbit.uvm.seq_items import VectorItem
        from torchbit.core.vector import Vector

        vec = Vector.from_array(torch.tensor([1.0], dtype=torch.float32))
        item = VectorItem("test", vec)
        assert "VectorItem" in repr(item)

    def test_logic_sequence_item(self):
        from torchbit.uvm.seq_items import LogicSequenceItem
        from torchbit.core.logic_sequence import LogicSequence

        seq = LogicSequence([0xDEAD, 0xBEEF])
        item = LogicSequenceItem("test", seq)
        assert item.name == "test"
        assert LogicSequenceItem("a", seq) == LogicSequenceItem(
            "b", LogicSequence([0xDEAD, 0xBEEF])
        )


class TestDriverMonitorAgent:
    """Test non-UVM wrappers (no pyuvm needed)."""

    def test_torchbit_driver_init(self):
        from torchbit.uvm.driver import TorchbitDriver
        from torchbit.tools.strategy import RandomBackpressure

        class MockBFM:
            pass

        bfm = MockBFM()
        s = RandomBackpressure(0.5, seed=1)
        d = TorchbitDriver(bfm, strategy=s, debug=True)
        assert d.bfm is bfm
        assert d.strategy is s
        assert d.debug is True

    def test_torchbit_monitor_init(self):
        from torchbit.uvm.monitor import TorchbitMonitor

        class MockBFM:
            pass

        m = TorchbitMonitor(MockBFM(), debug=True)
        assert m.debug is True


class TestUvmImports:
    """Verify the uvm package imports work."""

    def test_import_uvm_package(self):
        from torchbit.uvm import TorchbitBFM, VectorItem, LogicSequenceItem
        from torchbit.uvm import TorchbitDriver, TorchbitMonitor, TorchbitAgent

        assert TorchbitBFM is not None
