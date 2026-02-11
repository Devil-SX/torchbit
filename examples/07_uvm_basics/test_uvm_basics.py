"""
UVM Basics Example

Demonstrates the basic UVM components from torchbit.uvm.
All components work as pure Python — no cocotb or simulator needed.

Components covered:
- TorchbitBFM: Signal bridge (InputPort/OutputPort)
- VectorItem / LogicSequenceItem: Sequence items
- TorchbitScoreboard: Expected vs actual comparison
- TorchbitDriver / TorchbitMonitor: Lightweight wrappers
- TorchbitAgent: Documentation-level agent
- TorchbitEnv / TorchbitTest: Documentation-level environment/test
"""
import torch
from torchbit.core import Vector
from torchbit.core.logic_sequence import LogicSequence
from torchbit.uvm import (
    TorchbitBFM,
    VectorItem,
    LogicSequenceItem,
    TorchbitScoreboard,
    TorchbitDriver,
    TorchbitMonitor,
    TorchbitAgent,
    TorchbitEnv,
    TorchbitTest,
)
from torchbit.tools.strategy import GreedyStrategy


def test_bfm():
    """Demo: TorchbitBFM as a signal bridge."""
    print("=" * 60)
    print("1. TorchbitBFM - Signal Bridge")
    print("=" * 60)
    print()

    # BFM wraps InputPort/OutputPort for pyuvm's ConfigDB pattern.
    # In a real test, dut and clk come from cocotb.
    # Here we pass None to show the API structure.
    bfm = TorchbitBFM(dut=None, clk=None)

    # Register signals (in a real test these would be cocotb signal handles)
    print("  BFM created with dut=None, clk=None (pure Python demo)")
    print(f"  has_input('data'): {bfm.has_input('data')}")
    print(f"  has_output('valid'): {bfm.has_output('valid')}")
    print()

    # BFM stores input/output registrations
    print("  In a real cocotb test:")
    print("    bfm = TorchbitBFM(dut=dut, clk=dut.clk)")
    print("    bfm.add_input('data', dut.data_in)")
    print("    bfm.add_output('valid', dut.valid_out)")
    print("    bfm.drive('data', 0xDEAD)")
    print("    val = bfm.sample('valid')")
    print()
    print("  PASSED")
    print()


def test_sequence_items():
    """Demo: VectorItem and LogicSequenceItem."""
    print("=" * 60)
    print("2. Sequence Items - VectorItem & LogicSequenceItem")
    print("=" * 60)
    print()

    # VectorItem wraps a torchbit Vector
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    vec = Vector.from_array(tensor)
    item1 = VectorItem("stimulus_1", vec)
    print(f"  VectorItem: {item1}")

    # Create another with same data for comparison
    vec2 = Vector.from_array(tensor.clone())
    item2 = VectorItem("stimulus_2", vec2)
    print(f"  VectorItem: {item2}")
    print(f"  item1 == item2: {item1 == item2}")
    assert item1 == item2, "Equal VectorItems should match"
    print()

    # LogicSequenceItem wraps a LogicSequence (list of packed ints)
    logic_seq = LogicSequence([0xDEAD, 0xBEEF, 0xCAFE])
    ls_item = LogicSequenceItem("batch_1", logic_seq)
    print(f"  LogicSequenceItem: {ls_item}")

    ls_item2 = LogicSequenceItem("batch_2", LogicSequence([0xDEAD, 0xBEEF, 0xCAFE]))
    print(f"  ls_item == ls_item2: {ls_item == ls_item2}")
    assert ls_item == ls_item2, "Equal LogicSequenceItems should match"
    print()

    # None items
    empty_item = VectorItem("empty", None)
    print(f"  Empty VectorItem: {empty_item}")
    print()
    print("  PASSED")
    print()


def test_scoreboard():
    """Demo: TorchbitScoreboard for comparison."""
    print("=" * 60)
    print("3. TorchbitScoreboard - Expected vs Actual Comparison")
    print("=" * 60)
    print()

    sb = TorchbitScoreboard(name="my_scoreboard")

    # Add matching items
    for i in range(5):
        tensor = torch.tensor([float(i)], dtype=torch.float32)
        expected = VectorItem(f"exp_{i}", Vector.from_array(tensor))
        actual = VectorItem(f"act_{i}", Vector.from_array(tensor.clone()))
        sb.add_expected(expected)
        sb.add_actual(actual)

    print(f"  After 5 matching pairs:")
    print(f"    Matches: {sb.match_count}")
    print(f"    Mismatches: {sb.mismatch_count}")
    print(f"    Passed: {sb.passed}")
    assert sb.passed, "Scoreboard should pass with all matches"
    print()

    # Add a mismatched item
    sb.add_expected(VectorItem("exp_bad", Vector.from_array(torch.tensor([99.0]))))
    sb.add_actual(VectorItem("act_bad", Vector.from_array(torch.tensor([0.0]))))

    print(f"  After adding a mismatch:")
    print(f"    Matches: {sb.match_count}")
    print(f"    Mismatches: {sb.mismatch_count}")
    print(f"    Passed: {sb.passed}")
    assert not sb.passed, "Scoreboard should fail with mismatch"
    print()

    # Full report
    print("  Report:")
    for line in sb.report().split("\n"):
        print(f"    {line}")
    print()
    print("  PASSED")
    print()


def test_driver_monitor():
    """Demo: TorchbitDriver and TorchbitMonitor."""
    print("=" * 60)
    print("4. TorchbitDriver & TorchbitMonitor")
    print("=" * 60)
    print()

    # TorchbitDriver uses a BFM and TransferStrategy
    strategy = GreedyStrategy()
    driver = TorchbitDriver(bfm=None, strategy=strategy, debug=True)
    print(f"  Driver created with strategy: {type(driver.strategy).__name__}")
    print(f"  Driver debug: {driver.debug}")
    print(f"  Driver bfm: {driver.bfm}")
    print()

    # TorchbitMonitor samples signals via BFM
    monitor = TorchbitMonitor(bfm=None, debug=False)
    print(f"  Monitor created with debug: {monitor.debug}")
    print(f"  Monitor bfm: {monitor.bfm}")
    print()

    print("  In a real cocotb test:")
    print("    driver = TorchbitDriver(bfm, strategy=RandomBackpressure(0.3))")
    print("    monitor = TorchbitMonitor(bfm)")
    print()
    print("  PASSED")
    print()


def test_agent():
    """Demo: TorchbitAgent."""
    print("=" * 60)
    print("5. TorchbitAgent - Assembles Driver + Monitor")
    print("=" * 60)
    print()

    agent = TorchbitAgent()
    print(f"  Agent created: {type(agent).__name__}")
    print()
    print("  TorchbitAgent is a documentation-level class.")
    print("  For pyuvm integration, use create_uvm_agent():")
    print("    agent_cls = create_uvm_agent(driver_cls, monitor_cls)")
    print()
    print("  PASSED")
    print()


def test_env_and_test():
    """Demo: TorchbitEnv and TorchbitTest."""
    print("=" * 60)
    print("6. TorchbitEnv & TorchbitTest - Environment and Test")
    print("=" * 60)
    print()

    # TorchbitEnv assembles agents + scoreboard
    env = TorchbitEnv(name="my_env")
    print(f"  Env created: '{env.name}'")
    print(f"  Agents: {env.agents}")
    print(f"  Scoreboard: {env.scoreboard}")

    # Add a scoreboard
    sb = TorchbitScoreboard("env_scoreboard")
    env.set_scoreboard(sb)
    print(f"  After set_scoreboard: {env.scoreboard.name}")
    print()

    # TorchbitTest orchestrates environment setup
    test = TorchbitTest(name="my_test")
    print(f"  Test created: '{test.name}'")
    print(f"  Test env: {test.env}")
    print()

    print("  For pyuvm integration:")
    print("    env_cls = create_uvm_env(agent_configs=[{'name': 'agent0'}])")
    print("    test_cls = create_uvm_test(env_cls)")
    print()
    print("  PASSED")
    print()


def main():
    """Run all UVM basics demos."""
    print()
    print("*" * 60)
    print("TorchBit UVM Basics Example")
    print("*" * 60)
    print()
    print("This example demonstrates torchbit.uvm components")
    print("that work as pure Python (no cocotb/pyuvm needed).")
    print()

    test_bfm()
    test_sequence_items()
    test_scoreboard()
    test_driver_monitor()
    test_agent()
    test_env_and_test()

    print("*" * 60)
    print("All UVM basics demos passed!")
    print("*" * 60)
    print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
