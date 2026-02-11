"""
UVM Advanced Example

Demonstrates advanced UVM components from torchbit.uvm.
All components work as pure Python — no cocotb or simulator needed.

Components covered:
- ComponentRegistry: Factory-style type overrides
- CoveragePoint / CoverageGroup: Functional coverage tracking
- RegisterModel / RegisterBlock: Register Abstraction Layer with BitStruct
"""
import torch
from torchbit.core.bit_struct import BitStruct, BitField
from torchbit.tools import Buffer
from torchbit.uvm import (
    ComponentRegistry,
    CoverageGroup,
    CoveragePoint,
    RegisterModel,
    RegisterBlock,
)


def test_component_registry():
    """Demo: ComponentRegistry for factory-style type overrides."""
    print("=" * 60)
    print("1. ComponentRegistry - Factory-Style Type Overrides")
    print("=" * 60)
    print()

    # Start clean
    ComponentRegistry.clear()

    # Check defaults
    print("  Default state:")
    print(f"    has_override('driver'): {ComponentRegistry.has_override('driver')}")
    default = ComponentRegistry.get("driver", default="DefaultDriver")
    print(f"    get('driver', default='DefaultDriver'): {default}")
    print()

    # Register an override
    class CustomDriver:
        """A custom driver class for testing."""
        pass

    ComponentRegistry.set_override("driver", CustomDriver)
    print("  After set_override('driver', CustomDriver):")
    print(f"    has_override('driver'): {ComponentRegistry.has_override('driver')}")
    result = ComponentRegistry.get("driver")
    print(f"    get('driver'): {result.__name__}")
    assert result is CustomDriver
    print()

    # Override with another class
    class DebugDriver:
        pass

    ComponentRegistry.set_override("driver", DebugDriver)
    result = ComponentRegistry.get("driver")
    print(f"  After second override: get('driver'): {result.__name__}")
    assert result is DebugDriver
    print()

    # Register multiple component types
    class CustomMonitor:
        pass

    ComponentRegistry.set_override("monitor", CustomMonitor)
    print(f"  Multiple overrides:")
    print(f"    get('driver'): {ComponentRegistry.get('driver').__name__}")
    print(f"    get('monitor'): {ComponentRegistry.get('monitor').__name__}")
    print(f"    get('scoreboard'): {ComponentRegistry.get('scoreboard', default='None')}")
    print()

    # Clear all
    ComponentRegistry.clear()
    print(f"  After clear:")
    print(f"    has_override('driver'): {ComponentRegistry.has_override('driver')}")
    print()
    print("  PASSED")
    print()


def test_coverage():
    """Demo: CoveragePoint and CoverageGroup for functional coverage."""
    print("=" * 60)
    print("2. CoveragePoint / CoverageGroup - Functional Coverage")
    print("=" * 60)
    print()

    # Create a coverage group for data values
    cg = CoverageGroup("data_values")

    # Add coverage points with bins
    cg.add_point("sign", [
        ("zero", lambda v: v == 0),
        ("positive", lambda v: v > 0),
        ("negative", lambda v: v < 0),
    ])

    cg.add_point("magnitude", [
        ("small", lambda v: 0 <= abs(v) < 10),
        ("medium", lambda v: 10 <= abs(v) < 100),
        ("large", lambda v: 100 <= abs(v) < 1000),
        ("huge", lambda v: abs(v) >= 1000),
    ])

    print("  Coverage group: 'data_values'")
    print(f"  Points: {list(cg.points.keys())}")
    print(f"  Initial coverage: {cg.coverage_pct:.1f}%")
    print()

    # Sample some values
    test_values = [0, 5, -3, 42, -100, 500, 2000]
    for val in test_values:
        cg.sample("sign", val)
        cg.sample("magnitude", val)

    print(f"  After sampling {test_values}:")
    print(f"  Coverage: {cg.coverage_pct:.1f}%")
    print(f"  All covered: {cg.covered}")
    print()

    # Print report
    print("  Report:")
    for line in cg.report().split("\n"):
        print(f"    {line}")
    print()

    # Check individual points
    sign_point = cg.points["sign"]
    print(f"  'sign' point coverage: {sign_point.coverage_pct:.1f}%")
    print(f"  'sign' hit counts: {sign_point.hit_counts}")
    assert sign_point.covered, "'sign' should be fully covered"
    print()

    # Standalone CoveragePoint
    print("  Standalone CoveragePoint:")
    cp = CoveragePoint("opcode", [
        ("NOP", lambda v: v == 0x00),
        ("ADD", lambda v: v == 0x01),
        ("SUB", lambda v: v == 0x02),
        ("MUL", lambda v: v == 0x03),
    ])
    cp.sample(0x00)
    cp.sample(0x01)
    print(f"    Coverage: {cp.coverage_pct:.1f}% (2/4 bins hit)")
    assert not cp.covered, "Should not be fully covered yet"
    cp.sample(0x02)
    cp.sample(0x03)
    print(f"    Coverage: {cp.coverage_pct:.1f}% (4/4 bins hit)")
    assert cp.covered, "Should now be fully covered"
    print()
    print("  PASSED")
    print()


def test_register_model():
    """Demo: RegisterModel for field-level register access."""
    print("=" * 60)
    print("3. RegisterModel - Register with BitStruct Fields")
    print("=" * 60)
    print()

    # Define a control register with BitStruct
    ctrl_struct = BitStruct([
        BitField("enable", 1),
        BitField("mode", 3),
        BitField("addr", 12),
        BitField("data", 16),
    ], lsb_first=True)

    model = RegisterModel("ctrl_reg", ctrl_struct, base_addr=0x00)

    print(f"  Register: '{model.name}' at addr {model.base_addr:#x}")
    print(f"  Fields: {model.fields}")
    print()

    # Write individual fields
    model.write_field("enable", 1)
    model.write_field("mode", 5)
    model.write_field("addr", 0x123)
    model.write_field("data", 0xABCD)

    print("  After writing fields:")
    print(f"    enable = {model.read_field('enable')}")
    print(f"    mode   = {model.read_field('mode')}")
    print(f"    addr   = {model.read_field('addr'):#05x}")
    print(f"    data   = {model.read_field('data'):#06x}")
    assert model.read_field("enable") == 1
    assert model.read_field("mode") == 5
    assert model.read_field("addr") == 0x123
    assert model.read_field("data") == 0xABCD
    print()

    # Get/set packed value
    packed = model.get_packed()
    print(f"  Packed value: {packed:#010x}")

    model.reset()
    print(f"  After reset: enable={model.read_field('enable')}, packed={model.get_packed():#x}")

    model.set_packed(packed)
    print(f"  After set_packed: enable={model.read_field('enable')}, data={model.read_field('data'):#06x}")
    assert model.read_field("data") == 0xABCD
    print()
    print("  PASSED")
    print()


def test_register_block():
    """Demo: RegisterBlock for multiple registers with backdoor access."""
    print("=" * 60)
    print("4. RegisterBlock - Register Bank with Buffer Backdoor")
    print("=" * 60)
    print()

    # Define register structs
    status_struct = BitStruct([
        BitField("busy", 1),
        BitField("error", 1),
        BitField("count", 14),
        BitField("reserved", 16),
    ], lsb_first=True)

    config_struct = BitStruct([
        BitField("enable", 1),
        BitField("mode", 3),
        BitField("threshold", 12),
        BitField("reserved", 16),
    ], lsb_first=True)

    # Create register block
    block = RegisterBlock("ctrl", base_addr=0x10)
    block.add_register("status", status_struct, offset=0)
    block.add_register("config", config_struct, offset=1)

    print(f"  Block: '{block.name}' at base {block.base_addr:#x}")
    print(f"  Registers: {list(block.registers.keys())}")
    print()

    # Write via block
    block.write("config", "enable", 1)
    block.write("config", "mode", 7)
    block.write("config", "threshold", 0x3FF)
    block.write("status", "busy", 1)
    block.write("status", "count", 42)

    print("  After writing:")
    print(f"    config.enable    = {block.read('config', 'enable')}")
    print(f"    config.mode      = {block.read('config', 'mode')}")
    print(f"    config.threshold = {block.read('config', 'threshold'):#05x}")
    print(f"    status.busy      = {block.read('status', 'busy')}")
    print(f"    status.count     = {block.read('status', 'count')}")
    print()

    # Backdoor access via Buffer
    buf = Buffer(width=32, depth=256)

    # Write register values into buffer
    block.backdoor_write(buf, "config")
    block.backdoor_write(buf, "status")

    config_addr = block.get_register("config").base_addr
    status_addr = block.get_register("status").base_addr
    print(f"  Buffer backdoor write:")
    print(f"    buf[{config_addr:#x}] = {buf.read(config_addr):#010x} (config)")
    print(f"    buf[{status_addr:#x}] = {buf.read(status_addr):#010x} (status)")
    print()

    # Modify buffer and read back
    block.write("config", "enable", 0)
    block.write("config", "mode", 0)
    print(f"  After clearing config fields: enable={block.read('config', 'enable')}")

    # Read from buffer to restore
    block.backdoor_read(buf, "config")
    print(f"  After backdoor_read from buffer: enable={block.read('config', 'enable')}, mode={block.read('config', 'mode')}")
    assert block.read("config", "enable") == 1
    assert block.read("config", "mode") == 7
    print()
    print("  PASSED")
    print()


def main():
    """Run all UVM advanced demos."""
    print()
    print("*" * 60)
    print("TorchBit UVM Advanced Example")
    print("*" * 60)
    print()
    print("This example demonstrates advanced torchbit.uvm components")
    print("that work as pure Python (no cocotb/pyuvm needed).")
    print()

    test_component_registry()
    test_coverage()
    test_register_model()
    test_register_block()

    print("*" * 60)
    print("All UVM advanced demos passed!")
    print("*" * 60)
    print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
