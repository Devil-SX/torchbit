# Verification API Reference

Torchbit provides two API modes for hardware verification:

- **Native torchbit** — direct cocotb integration; lightweight, no extra dependencies.
- **PyUVM-compatible** — factory functions that generate `pyuvm` component classes; follows UVM methodology.

Both share the same underlying data types (`Vector`, `LogicSequence`, `TileMapping`, etc.) and can be mixed in one project.

## Table of Contents

- [API Comparison Table](#api-comparison-table)
- [Native Torchbit API](#native-torchbit-api)
  - [Signal Ports](#signal-ports)
  - [Driver](#driver)
  - [FIFODriver](#fifodriver)
  - [PoolMonitor](#poolmonitor)
  - [FIFOMonitor](#fifomonitor)
  - [FIFOReceiver](#fiforeceiver)
  - [Buffer / TwoPortBuffer](#buffer--twoportbuffer)
  - [TransferStrategy](#transferstrategy)
  - [ComponentDB](#componentdb)
  - [Native Workflow Example](#native-workflow-example)
- [PyUVM-Compatible API](#pyuvm-compatible-api)
  - [TorchbitBFM](#torchbitbfm)
  - [Sequence Items](#sequence-items)
  - [TorchbitScoreboard](#torchbitscoreboard)
  - [Factory Functions](#factory-functions)
  - [ComponentRegistry](#componentregistry)
  - [Coverage](#coverage)
  - [Register Abstraction Layer (RAL)](#register-abstraction-layer-ral)
  - [PyUVM Workflow Example](#pyuvm-workflow-example)
- [Choosing Between the Two APIs](#choosing-between-the-two-apis)

---

## API Comparison Table

| Verification Concept | Native Torchbit (`torchbit.tools`) | PyUVM-Compatible (`torchbit.uvm`) | Requires pyuvm |
|---|---|---|:---:|
| **Signal abstraction** | `InputPort` / `OutputPort` | `TorchbitBFM` | No |
| **Drive stimulus** | `Driver` / `FIFODriver` | `create_uvm_driver()` | Yes |
| **Capture response** | `PoolMonitor` / `FIFOMonitor` / `FIFOReceiver` | `create_uvm_monitor()` | Yes |
| **Result comparison** | manual `torch.equal()` | `TorchbitScoreboard` | No |
| **Sequence item** | `LogicSequence` (plain list) | `VectorItem` / `LogicSequenceItem` | No |
| **Agent** | *(no equivalent — compose manually)* | `create_uvm_agent()` | Yes |
| **Environment** | *(no equivalent)* | `TorchbitEnv` / `create_uvm_env()` | No / Yes |
| **Test** | `@cocotb.test()` function | `TorchbitTest` / `create_uvm_test()` | No / Yes |
| **Transfer timing** | `TransferStrategy` subclasses | same (`TransferStrategy`) | No |
| **Memory model** | `Buffer` / `TwoPortBuffer` | same | No |
| **Component discovery** | `ComponentDB` (path-based) | `ComponentRegistry` (type override) | No |
| **Functional coverage** | *(no equivalent)* | `CoverageGroup` / `CoveragePoint` | No |
| **Register model** | *(no equivalent)* | `RegisterModel` / `RegisterBlock` | No |

---

## Native Torchbit API

All classes live in `torchbit.tools` and work directly with cocotb — no pyuvm dependency.

### Signal Ports

```python
from torchbit.tools import InputPort, OutputPort
```

| Class | Purpose | Key method |
|-------|---------|------------|
| `InputPort(signal)` | Read a DUT signal | `get() -> int` |
| `OutputPort(signal, set_immediately=False)` | Drive a DUT signal | `set(value: int)` |

Both handle `None` gracefully (return 0 / no-op).

### Driver

```python
from torchbit.tools import Driver
```

The original stimulus driver. Sends a `LogicSequence` through `data` + `valid` signals with optional `full` backpressure.

```python
driver = Driver(debug=False)
driver.connect(dut, clk, data=dut.din, valid=dut.din_vld, full=None)
driver.load([0x1, 0x2, 0x3])
await driver.run()
```

| Method | Signature |
|--------|-----------|
| `connect` | `(dut, clk, data, valid, full=None)` |
| `load` | `(sequence: LogicSequence)` |
| `run` | `async (stop_event=None)` |
| `dump_time` | `() -> List[tuple]` |

### FIFODriver

```python
from torchbit.tools import FIFODriver
```

Enhanced driver with pluggable `TransferStrategy` and explicit polarity control (`active_high`).

```python
driver = FIFODriver(strategy=RandomBackpressure(0.3))
driver.connect(dut, clk, data=dut.din, valid=dut.din_vld, ready=dut.din_rdy)
driver.load([0x1, 0x2, 0x3])
await driver.run()
```

Supports `ComponentDB` path-based creation:

```python
driver = FIFODriver.from_path("top.encoder.input", dut, dut.clk)
```

### PoolMonitor

```python
from torchbit.tools import PoolMonitor
```

Always-ready receiver. Captures data when `valid` is asserted.

```python
monitor = PoolMonitor(debug=False)
monitor.connect(dut, clk, data=dut.dout, valid=dut.dout_vld)

stop_event = Event()
cocotb.start_soon(monitor.run(stop_event))
# ... run simulation ...
stop_event.set()
results = monitor.dump()          # LogicSequence
```

### FIFOMonitor

```python
from torchbit.tools import FIFOMonitor
```

Receiver with FIFO-style flow control. Drives `ready` based on `empty` signal.

```python
monitor = FIFOMonitor(debug=False)
monitor.connect(dut, clk, data=dut.dout, empty=dut.fifo_empty,
                ready=dut.fifo_ready, valid=dut.dout_vld)
```

### FIFOReceiver

```python
from torchbit.tools import FIFOReceiver
```

Enhanced receiver with `TransferStrategy`-based `ready` assertion. Replaces `FIFOMonitor` for new designs.

```python
receiver = FIFOReceiver(strategy=BurstStrategy(burst_len=8, pause_cycles=4))
receiver.connect(dut, clk, data=dut.dout, valid=dut.dout_vld, ready=dut.dout_rdy)
```

### Buffer / TwoPortBuffer

```python
from torchbit.tools import Buffer, TwoPortBuffer
```

In-memory models with backdoor access. `TwoPortBuffer` adds dual-port HDL connectivity.

```python
buf = TwoPortBuffer(width=32, depth=256, backpressure=False)
buf.connect(dut, clk,
            wr_csb=dut.wr_csb, wr_din=dut.wr_din, wr_addr=dut.wr_addr,
            rd_csb=dut.rd_csb, rd_addr=dut.rd_addr,
            rd_dout=dut.rd_dout, rd_dout_vld=dut.rd_dout_vld)
await buf.init()
cocotb.start_soon(buf.run())

# Backdoor access (instant, no clock)
buf.write(0x00, 0xDEADBEEF)
val = buf.read(0x00)

# Tensor backdoor (requires TileMapping + AddressMapping)
buf.backdoor_load_tensor(tensor, tile_mapping, addr_mapping)
result = buf.backdoor_dump_tensor(tile_mapping, addr_mapping)
```

### TransferStrategy

```python
from torchbit.tools import (
    GreedyStrategy, RandomBackpressure, BurstStrategy, ThrottledStrategy,
)
```

All strategies implement `should_transfer(cycle: int) -> bool`:

| Strategy | Constructor | Behavior |
|----------|-------------|----------|
| `GreedyStrategy` | `()` | Always transfer |
| `RandomBackpressure` | `(stall_prob=0.3, seed=None)` | Stall with probability |
| `BurstStrategy` | `(burst_len=8, pause_cycles=4)` | N-on, M-off pattern |
| `ThrottledStrategy` | `(min_interval=3)` | One per N cycles |

### ComponentDB

```python
from torchbit.tools import ComponentDB
```

Hierarchical path-based signal registry. Register once, resolve anywhere.

```python
ComponentDB.set("top.encoder.input", {
    "data": dut.din, "valid": dut.din_vld, "ready": dut.din_rdy,
})
driver = FIFODriver.from_path("top.encoder.input", dut, dut.clk)
```

### Native Workflow Example

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Event
from torchbit.tools import Driver, PoolMonitor

@cocotb.test()
async def test_pipe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Drive
    driver = Driver(debug=True)
    driver.connect(dut, dut.clk, dut.din, dut.din_vld, full=None)
    driver.load([10, 20, 30])

    # Monitor
    monitor = PoolMonitor()
    monitor.connect(dut, dut.clk, dut.dout, dut.dout_vld)
    stop_event = Event()
    cocotb.start_soon(monitor.run(stop_event))

    await driver.run()

    for _ in range(10):
        await RisingEdge(dut.clk)
    stop_event.set()

    assert list(monitor.dump()) == [10, 20, 30]
```

---

## PyUVM-Compatible API

Components in `torchbit.uvm`. The **base classes** (`TorchbitBFM`, `TorchbitScoreboard`, `VectorItem`, etc.) work as pure Python — no pyuvm needed. The **factory functions** (`create_uvm_driver`, `create_uvm_agent`, etc.) generate `pyuvm` subclasses and require `pyuvm >= 4.0.0`.

### TorchbitBFM

```python
from torchbit.uvm import TorchbitBFM
```

Signal abstraction layer — registers DUT signals by name, drives/samples through a uniform interface.

```python
bfm = TorchbitBFM(dut=dut, clk=dut.clk)
bfm.add_input('din', dut.din)           # TB drives this signal
bfm.add_input('din_vld', dut.din_vld)
bfm.add_output('dout', dut.dout)         # TB samples this signal
bfm.add_output('dout_vld', dut.dout_vld)

bfm.drive('din', 0xBEEF)
bfm.drive('din_vld', 1)
val = bfm.sample('dout')
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_input` | `(name, signal)` | Register a DUT input (wraps in `OutputPort`) |
| `add_output` | `(name, signal)` | Register a DUT output (wraps in `InputPort`) |
| `drive` | `(name, value)` | Drive a registered input |
| `sample` | `(name) -> int` | Sample a registered output |
| `has_input` | `(name) -> bool` | Check registration |
| `has_output` | `(name) -> bool` | Check registration |

### Sequence Items

```python
from torchbit.uvm import VectorItem, LogicSequenceItem
```

Wrappers that give torchbit data types UVM sequence item semantics (named, comparable).

```python
from torchbit.core import Vector
import torch

item = VectorItem("stim_0", Vector.from_array(torch.tensor([1, 2, 3])))
other = VectorItem("stim_1", Vector.from_array(torch.tensor([1, 2, 3])))
assert item == other   # compares underlying tensor with torch.equal()
```

### TorchbitScoreboard

```python
from torchbit.uvm import TorchbitScoreboard
```

Expected-vs-actual comparison engine. Works as pure Python.

```python
sb = TorchbitScoreboard(name="my_scoreboard")

sb.add_expected(VectorItem("exp", vec_a))
sb.add_actual(VectorItem("act", vec_b))      # triggers comparison immediately

print(sb.match_count, sb.mismatch_count)
assert sb.passed
print(sb.report())
```

| Method / Property | Description |
|---|---|
| `add_expected(item)` | Push expected item |
| `add_actual(item)` | Push actual item; triggers compare if expected available |
| `passed: bool` | `True` if no mismatches and all queues drained |
| `match_count: int` | Number of matches |
| `mismatch_count: int` | Number of mismatches |
| `report() -> str` | Formatted summary |

### Factory Functions

These generate `pyuvm` subclasses. Requires `pyuvm >= 4.0.0`.

```python
from torchbit.uvm.driver import create_uvm_driver
from torchbit.uvm.monitor import create_uvm_monitor
from torchbit.uvm.scoreboard import create_uvm_scoreboard
from torchbit.uvm.agent import create_uvm_agent
from torchbit.uvm.env import create_uvm_env
from torchbit.uvm.test import create_uvm_test
```

| Factory | Returns | Base class | Key phases |
|---------|---------|------------|------------|
| `create_uvm_driver(name)` | `uvm_driver` subclass | `pyuvm.uvm_driver` | `build_phase` (BFM + STRATEGY from ConfigDB), `run_phase` (drive loop) |
| `create_uvm_monitor(name)` | `uvm_monitor` subclass | `pyuvm.uvm_monitor` | `build_phase` (analysis port + BFM), `run_phase` (sample loop) |
| `create_uvm_scoreboard(name)` | `uvm_component` subclass | `pyuvm.uvm_component` | `build_phase` (FIFOs), `check_phase` (drain + assert) |
| `create_uvm_agent(driver_cls, monitor_cls, name)` | `uvm_agent` subclass | `pyuvm.uvm_agent` | `build_phase` (sequencer + driver + monitor), `connect_phase` |
| `create_uvm_env(agent_configs, name)` | `uvm_env` subclass | `pyuvm.uvm_env` | `build_phase` (agents + scoreboard), `connect_phase` |
| `create_uvm_test(env_cls, name)` | `uvm_test` subclass | `pyuvm.uvm_test` | `build_phase` (env) |

### ComponentRegistry

```python
from torchbit.uvm import ComponentRegistry
```

Type override registry (UVM factory pattern).

```python
ComponentRegistry.set_override("Driver", MyCustomDriver)
cls = ComponentRegistry.get("Driver", default=DefaultDriver)
```

### Coverage

```python
from torchbit.uvm import CoverageGroup, CoveragePoint
```

Functional coverage tracking with bins.

```python
cg = CoverageGroup("transfer_cov")
cg.add_point("size", [
    ("small",  lambda v: v < 16),
    ("medium", lambda v: 16 <= v < 64),
    ("large",  lambda v: v >= 64),
])
cg.sample("size", 32)
print(cg.coverage_pct, cg.report())
```

### Register Abstraction Layer (RAL)

```python
from torchbit.uvm import RegisterModel, RegisterBlock
```

Register modeling based on `BitStruct` definitions, with backdoor access through `Buffer`.

```python
block = RegisterBlock("ctrl", base_addr=0x1000)
reg = block.add_register("config", ConfigBitStruct, offset=0)
block.write("config", "enable", 1)
val = block.read("config", "enable")

# Backdoor access through Buffer
block.backdoor_write(buffer, "config")
block.backdoor_read(buffer, "config")
```

### PyUVM Workflow Example

Using BFM + Scoreboard with cocotb (no pyuvm dependency):

```python
import cocotb
import torch
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Event
from torchbit.core import Vector
from torchbit.uvm import TorchbitBFM, TorchbitScoreboard, VectorItem
from torchbit.tools.strategy import GreedyStrategy

@cocotb.test()
async def test_pipe_uvm(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # BFM: register signals
    bfm = TorchbitBFM(dut=dut, clk=dut.clk)
    bfm.add_input('din', dut.din)
    bfm.add_input('din_vld', dut.din_vld)
    bfm.add_output('dout', dut.dout)
    bfm.add_output('dout_vld', dut.dout_vld)

    # Scoreboard: load expected
    sb = TorchbitScoreboard(name="sb")
    sequence = [10, 20, 30]
    for v in sequence:
        sb.add_expected(VectorItem(f"e_{v}",
            Vector.from_array(torch.tensor([v], dtype=torch.int32))))

    # Monitor task
    stop_event = Event()
    async def monitor():
        while not stop_event.is_set():
            await RisingEdge(dut.clk)
            if int(bfm.sample('dout_vld')) == 1:
                val = int(bfm.sample('dout'))
                sb.add_actual(VectorItem(f"a_{val}",
                    Vector.from_array(torch.tensor([val], dtype=torch.int32))))
    cocotb.start_soon(monitor())

    # Drive through BFM
    for v in sequence:
        bfm.drive('din', v)
        bfm.drive('din_vld', 1)
        await RisingEdge(dut.clk)
    bfm.drive('din_vld', 0)

    await ClockCycles(dut.clk, 10)
    stop_event.set()

    assert sb.passed, sb.report()
```

Full pyuvm factory workflow (requires `pyuvm >= 4.0.0`):

```python
from torchbit.uvm.driver import create_uvm_driver
from torchbit.uvm.monitor import create_uvm_monitor
from torchbit.uvm.agent import create_uvm_agent
from torchbit.uvm.env import create_uvm_env
from torchbit.uvm.test import create_uvm_test

DriverCls = create_uvm_driver()
MonitorCls = create_uvm_monitor()
AgentCls = create_uvm_agent(DriverCls, MonitorCls)
EnvCls = create_uvm_env(agent_configs=[{"name": "pipe_agent"}])
TestCls = create_uvm_test(env_cls=EnvCls)
```

---

## Choosing Between the Two APIs

| Consideration | Native torchbit | PyUVM-compatible |
|---|---|---|
| **Dependencies** | cocotb only | cocotb + pyuvm |
| **Learning curve** | Lower — plain async functions | Higher — UVM phases, ConfigDB, sequencers |
| **Reusability** | Manual composition | Systematic (agent/env/test hierarchy) |
| **Scoreboard** | DIY with `torch.equal` | Built-in `TorchbitScoreboard` |
| **Coverage** | Not available | `CoverageGroup` / `CoveragePoint` |
| **Register model** | Not available | `RegisterModel` / `RegisterBlock` |
| **Best for** | Small modules, quick prototyping | Large designs, multi-agent environments |

**Mixing is encouraged.** A typical middle-ground approach:

- Use `TorchbitBFM` for signal abstraction (no pyuvm needed).
- Use `TorchbitScoreboard` + `VectorItem` for result checking (no pyuvm needed).
- Use native `TwoPortBuffer` for memory modeling.
- Use `TransferStrategy` for timing control.
- Skip the pyuvm factory functions unless you need full UVM phase machinery.

See `examples/09_uvm_pipeline/` for a working example of this mixed approach.
