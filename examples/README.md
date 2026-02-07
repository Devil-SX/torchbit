# TorchBit Examples

This directory contains examples demonstrating key TorchBit functionality for hardware verification with PyTorch.

## Prerequisites

- **Conda environment**: `torch2_9`
- **Simulator**: Verilator (default) or VCS
- **Python**: Python 3.10+

## Examples Overview

| Example | Description | DUT | Key Classes |
|---------|-------------|-----|-------------|
| [01_basic_runner](./01_basic_runner/) | Minimal Runner setup | `simple_counter.v` | Runner, FileConfig, BuildConfig |
| [02_data_convert](./02_data_convert/) | Vector/Matrix conversion | None | Vector, Matrix |
| [03_bit_struct](./03_bit_struct/) | BitField/BitStruct usage | None | BitStruct, BitField |
| [04_driver_monitor](./04_driver_monitor/) | Driver/Monitor pattern | `pipe.sv` | Driver, PoolMonitor |
| [05_buffer](./05_buffer/) | TwoPortBuffer with memory | `memory_mover.v` + `twopr_ram.v` | TwoPortBuffer, TileMapping |

## Running an Example

Each example has a `run.py` entry point:

```bash
cd examples/01_basic_runner
python run.py
```

## Example Details

### 1. Basic Runner
Demonstrates minimal Runner setup for compiling and running a simple DUT.
- Shows FileConfig and BuildConfig usage
- Compiles with Verilator and generates waveforms
- No complex verification - just compile/run

### 2. Data Convert
Pure Python example showing data conversion utilities.
- **Vector**: 1D tensor ↔ integer for HDL interfaces
- **Matrix**: 2D tensor ↔ files (hex/bin) for memory
- All supported dtypes: float16, bfloat16, float32, int8, int32, uint8, etc.

### 3. BitStruct
Demonstrates bit-level data manipulation.
- **BitField**: Named bit fields
- **BitStruct**: Packed data structures
- Useful for: CPU registers, instruction formats, protocol headers

### 4. Driver/Monitor
Shows stimulus generation and response capture.
- **Driver**: Drives data to DUT inputs
- **PoolMonitor**: Captures data from DUT outputs
- Generates timing visualization graphs

### 5. Buffer
Demonstrates TwoPortBuffer for backdoor memory access (chip_verification skill format).
- **TwoPortBuffer**: Backdoor read/write to hardware memory
- **TileMapping**: Address translation for tensor layouts
- **Testframework**: Wrapper + Golden Model pattern
- Follows chip_verification skill structure with tb/ directory

## File Structure

```
examples/
├── README.md                    # This file
├── common/                      # Shared utilities
│   └── __init__.py
├── 01_basic_runner/
├── 02_data_convert/
├── 03_bit_struct/
├── 04_driver_monitor/
└── 05_buffer/
```

## Viewing Waveforms

After running an example, waveforms are available in the build directory:

```bash
gtkwave sim_<name>_default_verilator_test_<module>/waveform.fst
```

## Simulator Configuration

By default, examples use **Verilator**. To use VCS instead:

```python
from torchbit.runner import BuildConfig, DEFAULT_VCS_BUILD_CONFIG

build_config = DEFAULT_VCS_BUILD_CONFIG
```

## Next Steps

After running these examples, explore:
- `tests/` - Formal pytest test suite
- `torchbit/core/` - Vector and Matrix implementation
- `torchbit/tools/` - Driver, Monitor, Buffer classes
- `torchbit/runner/` - Runner configuration
