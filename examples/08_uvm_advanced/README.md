# Example 08: UVM Advanced

Demonstrates advanced UVM components from `torchbit.uvm`. All components work as pure Python — no cocotb or simulator needed.

## Components

| Component | Description |
|-----------|-------------|
| `ComponentRegistry` | Factory-style type overrides for test customization |
| `CoveragePoint` | Tracks bin hits for a single coverage point |
| `CoverageGroup` | Groups multiple coverage points with reporting |
| `RegisterModel` | Maps BitStruct fields to an addressable register |
| `RegisterBlock` | Container for multiple registers with Buffer backdoor |

## Running

```bash
cd examples/08_uvm_advanced
python run.py
```

## Key Concepts

### Factory Overrides

```python
from torchbit.uvm import ComponentRegistry

class CustomDriver:
    pass

ComponentRegistry.set_override("driver", CustomDriver)
cls = ComponentRegistry.get("driver", default=DefaultDriver)
# cls is CustomDriver
```

### Functional Coverage

```python
from torchbit.uvm import CoverageGroup

cg = CoverageGroup("data_range")
cg.add_point("value", [
    ("zero", lambda v: v == 0),
    ("positive", lambda v: v > 0),
    ("negative", lambda v: v < 0),
])
cg.sample("value", 42)
print(cg.report())
```

### Register Abstraction Layer (RAL)

```python
from torchbit.core.bit_struct import BitStruct, BitField
from torchbit.uvm import RegisterModel, RegisterBlock
from torchbit.tools import Buffer

fields = [BitField("enable", 1), BitField("mode", 3), BitField("data", 28)]
struct_cls = BitStruct(fields, lsb_first=True)

block = RegisterBlock("ctrl", base_addr=0x10)
block.add_register("config", struct_cls, offset=0)
block.write("config", "enable", 1)

# Backdoor access via Buffer
buf = Buffer(width=32, depth=256)
block.backdoor_write(buf, "config")
```
