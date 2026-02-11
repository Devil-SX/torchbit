"""Tests for torchbit.uvm advanced features (no pyuvm needed)."""
import pytest
from torchbit.uvm.factory import ComponentRegistry
from torchbit.uvm.coverage import CoverageGroup, CoveragePoint
from torchbit.uvm.ral import RegisterModel, RegisterBlock
from torchbit.core.bit_struct import BitStruct, BitField
from torchbit.tools.buffer import Buffer


class TestComponentRegistry:
    def setup_method(self):
        ComponentRegistry.clear()

    def test_set_and_get_override(self):
        ComponentRegistry.set_override("driver", "MockDriver")
        assert ComponentRegistry.get("driver") == "MockDriver"

    def test_get_default(self):
        assert ComponentRegistry.get("driver", "DefaultDriver") == "DefaultDriver"

    def test_has_override(self):
        assert not ComponentRegistry.has_override("driver")
        ComponentRegistry.set_override("driver", "X")
        assert ComponentRegistry.has_override("driver")

    def test_clear(self):
        ComponentRegistry.set_override("driver", "X")
        ComponentRegistry.clear()
        assert not ComponentRegistry.has_override("driver")


class TestCoveragePoint:
    def test_basic_sampling(self):
        cp = CoveragePoint("val", [
            ("zero", lambda v: v == 0),
            ("positive", lambda v: v > 0),
        ])
        cp.sample(0)
        cp.sample(5)
        assert cp.covered
        assert cp.coverage_pct == 100.0

    def test_partial_coverage(self):
        cp = CoveragePoint("val", [
            ("a", lambda v: v == 1),
            ("b", lambda v: v == 2),
        ])
        cp.sample(1)
        assert not cp.covered
        assert cp.coverage_pct == 50.0

    def test_empty_bins(self):
        cp = CoveragePoint("empty", [])
        assert cp.covered
        assert cp.coverage_pct == 100.0


class TestCoverageGroup:
    def test_full_coverage(self):
        cg = CoverageGroup("test")
        cg.add_point("sign", [
            ("neg", lambda v: v < 0),
            ("pos", lambda v: v > 0),
        ])
        cg.sample("sign", -1)
        cg.sample("sign", 1)
        assert cg.covered

    def test_report(self):
        cg = CoverageGroup("data")
        cg.add_point("range", [
            ("low", lambda v: v < 10),
            ("high", lambda v: v >= 10),
        ])
        cg.sample("range", 5)
        report = cg.report()
        assert "data" in report
        assert "50.0%" in report

    def test_empty_group(self):
        cg = CoverageGroup("empty")
        assert cg.covered
        assert cg.coverage_pct == 100.0


class TestRegisterModel:
    def test_write_read_field(self):
        fields = [BitField("opcode", 8), BitField("addr", 16)]
        struct_cls = BitStruct(fields, lsb_first=True)
        reg = RegisterModel("ctrl", struct_cls)
        reg.write_field("opcode", 0x42)
        assert reg.read_field("opcode") == 0x42

    def test_packed_roundtrip(self):
        fields = [BitField("a", 8), BitField("b", 8)]
        struct_cls = BitStruct(fields, lsb_first=True)
        reg = RegisterModel("test", struct_cls)
        reg.write_field("a", 0xAB)
        reg.write_field("b", 0xCD)
        packed = reg.get_packed()
        reg.reset()
        assert reg.read_field("a") == 0
        reg.set_packed(packed)
        assert reg.read_field("a") == 0xAB
        assert reg.read_field("b") == 0xCD

    def test_fields_list(self):
        fields = [BitField("x", 4), BitField("y", 4)]
        struct_cls = BitStruct(fields, lsb_first=True)
        reg = RegisterModel("r", struct_cls)
        assert reg.fields == ["x", "y"]


class TestRegisterBlock:
    def test_add_and_access(self):
        fields = [BitField("en", 1), BitField("mode", 3)]
        struct_cls = BitStruct(fields, lsb_first=True)
        block = RegisterBlock("ctrl", base_addr=0x100)
        block.add_register("config", struct_cls, offset=0)
        block.write("config", "en", 1)
        block.write("config", "mode", 5)
        assert block.read("config", "en") == 1
        assert block.read("config", "mode") == 5

    def test_backdoor_roundtrip(self):
        fields = [BitField("val", 16)]
        struct_cls = BitStruct(fields, lsb_first=True)
        block = RegisterBlock("regs", base_addr=0)
        block.add_register("r0", struct_cls, offset=0)
        block.write("r0", "val", 0xBEEF)

        buf = Buffer(width=32, depth=16)
        block.backdoor_write(buf, "r0")

        block.write("r0", "val", 0)
        block.backdoor_read(buf, "r0")
        assert block.read("r0", "val") == 0xBEEF

    def test_register_address(self):
        fields = [BitField("data", 8)]
        struct_cls = BitStruct(fields, lsb_first=True)
        block = RegisterBlock("io", base_addr=0x200)
        reg = block.add_register("status", struct_cls, offset=4)
        assert reg.base_addr == 0x204
