# MemoryMoverBuffer Example

This example demonstrates the TwoPortBuffer class from torchbit for backdoor memory access in hardware verification, following the chip_verification skill format.

## Overview

The MemoryMover DUT copies data from a source memory region to a destination region. **TwoPortBuffer is used directly as the memory** - no separate Verilog RAM module is needed.

TwoPortBuffer provides:
- **Backdoor writes**: Initialize source data before DUT operation
- **Backdoor reads**: Verify destination data after DUT operation
- **Frontend access**: DUT reads/writes through the standard read/write ports

This approach avoids having to create a separate Verilog RAM module - TwoPortBuffer acts as the memory itself.

## Testframework Structure

```
05_buffer/
├── README.md                    # This file
├── run.py                       # Quick entry point for examples
├── src/rtl/                     # RTL source files
│   └── memory_mover.v           # DUT that copies data between regions
└── tb/                          # Testbench
    ├── common/                   # Shared components
    │   └── golden_model/         # Golden model (Python reference)
    └── dut_buffer/               # MemoryMover DUT tests
        ├── wrapper/              # DUT wrapper with TwoPortBuffer
        ├── tests/                # Test cases
        │   ├── test_copy_basic.py
        │   ├── test_copy_large.py
        │   ├── test_copy_overlap.py
        │   └── test_buffer_roundtrip.py
        ├── reports/              # Test reports (generated)
        └── main.py               # Entry point for running tests
```

## Key Concepts

### TwoPortBuffer

TwoPortBuffer provides **backdoor memory access** for tensor data:

```python
from torchbit.tools import TwoPortBuffer
import torch

# Create buffer
buffer = TwoPortBuffer(width=32, depth=256)

# Connect to DUT signals
buffer.connect(
    dut=dut,
    clk=dut.clk,
    wr_csb=dut.wr_csb,      # Write chip select (active low)
    wr_din=dut.wr_din,      # Write data input
    wr_addr=dut.wr_addr,    # Write address
    rd_csb=dut.rd_csb,      # Read chip select (active low)
    rd_addr=dut.rd_addr,    # Read address
    rd_dout=dut.rd_dout,    # Read data output
    rd_dout_vld=dut.rd_dout_vld  # Read data valid
)

# Backdoor write - load data directly into memory
data = torch.arange(16, dtype=torch.int32)
for i, value in enumerate(data):
    buffer.write(i, int(value.item()))

# Backdoor read - read data directly from memory
result = []
for i in range(16):
    result.append(buffer.read(i))
```

### Wrapper Pattern

The wrapper encapsulates:
- DUT and TwoPortBuffer connections
- Clock, reset, and control logic
- Clean interface aligned with golden model

```python
class MemoryMoverWrapper:
    def __init__(self, dut):
        self.buffer = TwoPortBuffer(width=32, depth=256)

    async def init(self):
        # Reset and startup
        ...

    def load_source_data(self, data, base_addr):
        # Backdoor write using TwoPortBuffer
        for i, value in enumerate(data):
            self.buffer.write(base_addr + i, int(value.item()))

    async def run_copy(self, src_base, dst_base, num_words):
        # Run DUT copy operation
        ...

    def read_destination_data(self, num_words, base_addr):
        # Backdoor read using TwoPortBuffer
        result = []
        for i in range(num_words):
            result.append(self.buffer.read(base_addr + i))
        return torch.tensor(result, dtype=torch.int32)
```

### Test Flow

1. **Initialize** wrapper (clock start, reset)
2. **Load source data** via TwoPortBuffer backdoor write
3. **Run DUT** copy operation (uses TwoPortBuffer frontend ports)
4. **Read destination data** via TwoPortBuffer backdoor read
5. **Compare** with golden model output

## Running Tests

### Quick run (using run.py)

```bash
cd examples/05_buffer
python run.py
```

### Full testbench (using main.py)

```bash
cd examples/05_buffer/tb/dut_buffer
python main.py                              # Run all tests
python main.py --test_case=test_copy_basic  # Run single test
```

## Test Cases

| Test | Description |
|------|-------------|
| `test_copy_basic` | Basic copy with 16 words |
| `test_copy_large` | Larger copy with 64 words |
| `test_copy_overlap` | Copy with overlapping regions |
| `test_buffer_roundtrip` | TwoPortBuffer write/read verification |

## DUT Description

### memory_mover.v

The DUT connects directly to TwoPortBuffer (torchbit's memory component):

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| clk | input | 1 | Clock |
| rst_n | input | 1 | Active-low reset |
| start | input | 1 | Start copy operation |
| src_base_addr | input | 8 | Source base address |
| dst_base_addr | input | 8 | Destination base address |
| num_words | input | 8 | Number of words to copy |
| done | output | 1 | Operation complete flag |
| src_rd_addr | output | 8 | TwoPortBuffer read address |
| src_rd_csb | output | 1 | TwoPortBuffer read chip select (active low) |
| src_rd_dout | input | 32 | TwoPortBuffer read data output |
| src_rd_dout_vld | input | 1 | TwoPortBuffer read data valid |
| dst_wr_addr | output | 8 | TwoPortBuffer write address |
| dst_wr_din | output | 32 | TwoPortBuffer write data input |
| dst_wr_csb | output | 1 | TwoPortBuffer write chip select (active low) |

**No separate RAM module needed** - TwoPortBuffer acts as the memory directly!

## Expected Output

After running tests:
- Build directory: `sim_memory_mover_default_verilator_tb.dut_buffer.tests.*`
- Waveform: `waveform.fst`
- Reports: `tb/dut_buffer/reports/`

## Key Differences from Other Examples

| Aspect | This Example | Other Examples |
|--------|-------------|----------------|
| Structure | Full testframework (tb/) | Simple files |
| Components | Wrapper + Golden Model | Direct cocotb tests |
| Memory access | TwoPortBuffer (backdoor + frontend) | Direct DUT interface |
| Entry point | main.py (with --test_case) | run.py |
| Verification | Golden model comparison | Direct assertions |
