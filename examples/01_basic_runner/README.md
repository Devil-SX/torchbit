# Basic Runner Example

This example demonstrates the minimal TorchBit Runner setup for compiling and running a simple DUT with cocotb.

## Overview

- **DUT**: `simple_counter.v` - A parameterized 8-bit counter
- **Test**: `test_simple_counter.py` - Minimal cocotb test (compile and run only)
- **Purpose**: Show how to set up Runner with FileConfig and BuildConfig

## DUT Description

The `simple_counter` module:
- Parameterized WIDTH (default: 8)
- Counts up when enabled
- Resets asynchronously (active-low reset)

### Ports

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| clk | input | 1 | Clock |
| rst_n | input | 1 | Active-low reset |
| enable | input | 1 | Enable counting |
| count | output | WIDTH | Current count value |

## Running the Example

```bash
cd examples/01_basic_runner
python run.py
```

## Expected Output

The script will:
1. Compile the Verilog design with Verilator
2. Run the cocotb test
3. Generate a waveform file (`waveform.fst`)

## Viewing the Waveform

To view the generated waveform:

```bash
gtkwave sim_basic_counter_default_verilator_test_simple_counter/waveform.fst
```

## File Structure

```
01_basic_runner/
├── README.md              # This file
├── run.py                 # Entry point script
├── dut/
│   └── simple_counter.v   # DUT source
└── test_simple_counter.py # Cocotb test module
```

## Key Concepts Demonstrated

1. **FileConfig**: Specify HDL source files and top module
2. **BuildConfig**: Configure simulator backend (Verilator)
3. **Runner**: Build and run tests
4. **Waveform Generation**: Automatic FST waveform creation
