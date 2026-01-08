# Sender/Collector Example

This example demonstrates the Sender and PoolCollector classes for driving stimulus and capturing response from a hardware DUT.

## Overview

- **DUT**: `pipe.sv` - Parameterized pipeline with configurable width and delay
- **Test**: `test_pipe.py` - Demonstrates Sender and PoolCollector usage
- **Purpose**: Show how to use Sender for driving inputs and PoolCollector for capturing outputs

## DUT Description

The `Pipe` module is a simple pipeline:
- **WIDTH** parameter (default: 32) - Data width
- **DELAY** parameter (default: 4) - Number of pipeline stages
- No backpressure/ready signals
- Valid signal accompanies data through pipeline

### Ports

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| clk | input | 1 | Clock |
| rst_n | input | 1 | Active-low reset |
| din_vld | input | 1 | Input data valid |
| din | input | WIDTH | Input data |
| dout_vld | output | 1 | Output data valid |
| dout | output | WIDTH | Output data |

## Classes Demonstrated

### Sender
```python
from torchbit.tools import Sender

sender = Sender(debug=True)
sender.connect(dut, clk, data, valid, full=None)  # full=None for no backpressure
sender.load([1, 2, 3, 4])  # Load data sequence
await sender.run()  # Send data
```

### PoolCollector
```python
from torchbit.tools import PoolCollector

collector = PoolCollector(debug=True)
collector.connect(dut, clk, data, valid)

stop_event = Event()
cocotb.start_soon(collector.run(stop_event))

# Later...
stop_event.set()
data = collector.dump()  # Get collected data
```

## Running the Example

```bash
cd examples/04_sender_collector
python run.py
```

## Expected Output

The script will:
1. Compile the pipeline DUT with Verilator
2. Run the cocotb tests
3. Generate a timing visualization graph (`pipe_timing.png`)
4. Generate a waveform file (`waveform.fst`)

## Test Cases

### test_pipe_basic
- Sends 20 random values through the pipeline
- Verifies data integrity
- Generates timing visualization

### test_pipe_single_value
- Sends a single value (42)
- Verifies it propagates correctly

## File Structure

```
04_sender_collector/
├── README.md         # This file
├── run.py            # Entry point script
├── dut/
│   └── pipe.sv       # Pipeline DUT
└── test_pipe.py      # Cocotb test with Sender/Collector
```

## Key Concepts Demonstrated

1. **Sender**: Driving stimulus with valid handshake
2. **PoolCollector**: Capturing output on valid assertion
3. **Timing Visualization**: Temporal event graphs
4. **Data Verification**: Comparing sent vs received data
5. **Wrapper Pattern**: Encapsulating DUT connections
