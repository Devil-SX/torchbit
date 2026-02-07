# Driver/Monitor Example

This example demonstrates the Driver and PoolMonitor classes for driving stimulus and capturing response from a hardware DUT.

## Overview

- **DUT**: `pipe.sv` - Parameterized pipeline with configurable width and delay
- **Test**: `test_pipe.py` - Demonstrates Driver and PoolMonitor usage
- **Purpose**: Show how to use Driver for driving inputs and PoolMonitor for capturing outputs

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

### Driver
```python
from torchbit.tools import Driver

driver = Driver(debug=True)
driver.connect(dut, clk, data, valid, full=None)  # full=None for no backpressure
driver.load([1, 2, 3, 4])  # Load data sequence
await driver.run()  # Send data
```

### PoolMonitor
```python
from torchbit.tools import PoolMonitor

monitor = PoolMonitor(debug=True)
monitor.connect(dut, clk, data, valid)

stop_event = Event()
cocotb.start_soon(monitor.run(stop_event))

# Later...
stop_event.set()
data = monitor.dump()  # Get collected data
```

## Running the Example

```bash
cd examples/04_driver_monitor
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
04_driver_monitor/
├── README.md         # This file
├── run.py            # Entry point script
├── dut/
│   └── pipe.sv       # Pipeline DUT
└── test_pipe.py      # Cocotb test with Driver/Monitor
```

## Key Concepts Demonstrated

1. **Driver**: Driving stimulus with valid handshake
2. **PoolMonitor**: Capturing output on valid assertion
3. **Timing Visualization**: Temporal event graphs
4. **Data Verification**: Comparing sent vs received data
5. **Wrapper Pattern**: Encapsulating DUT connections
