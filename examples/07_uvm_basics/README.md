# Example 07: UVM Basics

Demonstrates the basic UVM components from `torchbit.uvm`. All components work as pure Python — no cocotb or simulator needed.

## Components

| Component | Description |
|-----------|-------------|
| `TorchbitBFM` | Signal bridge wrapping InputPort/OutputPort |
| `VectorItem` | Sequence item wrapping a torchbit Vector |
| `LogicSequenceItem` | Sequence item wrapping a LogicSequence |
| `TorchbitScoreboard` | Expected vs actual comparison engine |
| `TorchbitDriver` | Driver using BFM + TransferStrategy |
| `TorchbitMonitor` | Monitor sampling signals via BFM |
| `TorchbitAgent` | Assembles driver + monitor + sequencer |
| `TorchbitEnv` | Verification environment (agents + scoreboard) |
| `TorchbitTest` | Test orchestration |

## Running

```bash
cd examples/07_uvm_basics
python run.py
```

## Key Concepts

### Standalone vs pyuvm

Each component has two forms:
- **Standalone** (e.g., `TorchbitScoreboard`): Works without pyuvm, pure Python
- **Factory** (e.g., `create_uvm_scoreboard()`): Returns a pyuvm-compatible class

### Scoreboard Pattern

```python
from torchbit.uvm import TorchbitScoreboard, VectorItem

sb = TorchbitScoreboard("my_sb")
sb.add_expected(VectorItem("exp", expected_vec))
sb.add_actual(VectorItem("act", actual_vec))
assert sb.passed
print(sb.report())
```
