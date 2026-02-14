# 09 — UVM Pipeline

UVM-style verification of a parameterized pipeline using torchbit UVM
components **with real RTL simulation**.

This example bridges examples 04 (Driver/Monitor with RTL) and 07/08
(UVM components in pure Python) by applying the UVM methodology to an
actual cocotb + Verilator simulation.

## DUT

`pipe.sv` — a shift-register pipeline that delays input data by `DELAY`
clock cycles without modification.

```
                 +-----------------+
  clk ─────────>|                 |
  rst_n ───────>|      Pipe       |
                |  WIDTH=32       |
  din[31:0] ───>|  DELAY=4        |───> dout[31:0]
  din_vld ─────>|                 |───> dout_vld
                 +-----------------+
```

## UVM Components Used

| Component           | Role                                             |
|---------------------|--------------------------------------------------|
| `TorchbitBFM`       | Signal abstraction — `drive()` / `sample()` via named ports |
| `TorchbitScoreboard`| Compares expected vs. actual `VectorItem` pairs  |
| `VectorItem`        | Sequence item wrapping a torchbit `Vector`        |
| `TransferStrategy`  | Controls driver timing (`GreedyStrategy`, `RandomBackpressure`) |
| `PipeGolden`        | Golden model — predicts output == input           |

## Testbench Topology

```
  ┌─────────────┐  drive_sequence()  ┌────────────┐  frontdoor   ┌────────┐
  │    Test      │ ─────────────────>│ TorchbitBFM │────din/vld──>│  Pipe  │
  │   (cocotb)   │                   │             │<──dout/vld───│  DUT   │
  │              │  monitor_output() │             │              │        │
  │              │ <─────────────────│             │              │        │
  └──────┬───────┘                   └────────────┘              └────────┘
         │                                  │
         │  load_expected()   ┌─────────────────────┐
         │                    │ TorchbitScoreboard   │
         └───────────────────>│  add_expected()      │
              add_actual() ──>│  add_actual()        │
              (from monitor)  │  passed / report()   │
                              └─────────────────────┘
         │
         │  predict()        ┌─────────────┐
         └──────────────────>│ PipeGolden   │
                             └─────────────┘
```

## Data Layout

- **TileMapping**: not used (streaming protocol, not memory-mapped)
- **AddressMapping**: not used

Data is driven/sampled as scalar `int32` values through the valid/data
interface on each clock cycle.

## Test Scenarios

| Test               | Purpose                          | Key Parameters          |
|--------------------|----------------------------------|-------------------------|
| `test_basic`       | Correctness with greedy driver   | 20 values, no stalls    |
| `test_backpressure`| Correctness under random stalls  | 30 values, 30% stall    |

## Running

```bash
# Quick run (basic test only)
conda run -n torch2_9 python run.py

# Full testframework
cd tb/dut_pipe
conda run -n torch2_9 python main.py

# Single test
conda run -n torch2_9 python main.py --test_case=test_backpressure
```

## Directory Structure

```
09_uvm_pipeline/
├── run.py                          Quick entry point
├── src/rtl/
│   └── pipe.sv                     DUT (parameterized pipeline)
└── tb/
    ├── common/golden_model/
    │   └── pipe_golden.py          Golden model
    └── dut_pipe/
        ├── main.py                 Full test runner
        ├── wrapper/
        │   └── dut_wrapper.py      BFM + Scoreboard wrapper
        └── tests/
            ├── test_basic.py       Greedy driver test
            └── test_backpressure.py Random stall test
```
