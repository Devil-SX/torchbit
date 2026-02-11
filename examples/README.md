# TorchBit Examples

This directory contains examples demonstrating key TorchBit functionality for hardware verification with PyTorch.

## Prerequisites

- **Conda environment**: `torch2_9`
- **Simulator**: Verilator (default) or VCS (for examples with DUT)
- **Python**: Python 3.10+

## Examples Overview

| Example | Description | DUT | Key Classes |
|---------|-------------|-----|-------------|
| [01_basic_runner](./01_basic_runner/) | Minimal Runner setup | `simple_counter.v` | Runner, FileConfig, BuildConfig |
| [02_data_convert](./02_data_convert/) | Vector/VectorSequence conversion | None | Vector, VectorSequence |
| [03_bit_struct](./03_bit_struct/) | BitField/BitStruct usage | None | BitStruct, BitField |
| [04_driver_monitor](./04_driver_monitor/) | Driver/Monitor pattern | `pipe.sv` | Driver, PoolMonitor |
| [05_buffer](./05_buffer/) | TwoPortBuffer with memory | `memory_mover.v` | TwoPortBuffer, TileMapping |
| [06_tile_mapping](./06_tile_mapping/) | TileMapping with spatial/temporal dims | `tile_mover.v` | TileMapping, AddressMapping |
| [07_uvm_basics](./07_uvm_basics/) | UVM basics (pure Python) | None | BFM, Scoreboard, Driver/Monitor |
| [08_uvm_advanced](./08_uvm_advanced/) | UVM advanced (pure Python) | None | ComponentRegistry, Coverage, RAL |

## Running an Example

Each example has a `run.py` entry point:

```bash
cd examples/01_basic_runner
python run.py
```

Pure Python examples (02, 03, 07, 08) can run without a simulator.

## Example Details

### 1. Basic Runner
Demonstrates minimal Runner setup for compiling and running a simple DUT.
- Shows FileConfig and BuildConfig usage
- Compiles with Verilator and generates waveforms
- No complex verification - just compile/run

### 2. Data Convert
Pure Python example showing data conversion utilities.
- **Vector**: 1D tensor ↔ integer for HDL interfaces
- **VectorSequence**: 2D tensor ↔ files (hex/bin) for memory
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

### 6. Tile Mapping
Demonstrates TileMapping with spatial/temporal dimension mapping.
- **TileMapping**: Tensor-to-memory layout with einops
- **AddressMapping**: Multi-dimensional to flat memory address translation
- **Spatial mapping**: cs elements transferred per cycle (parallel)
- Tile transpose: (b, w, c) → (w, b, c) layout

### 7. UVM Basics
Pure Python example showing basic UVM components from `torchbit.uvm`.
- **TorchbitBFM**: Signal bridge (InputPort/OutputPort)
- **VectorItem / LogicSequenceItem**: Sequence items
- **TorchbitScoreboard**: Expected vs actual comparison
- **TorchbitDriver / TorchbitMonitor / TorchbitAgent**: Lightweight wrappers

### 8. UVM Advanced
Pure Python example showing advanced UVM components from `torchbit.uvm`.
- **ComponentRegistry**: Factory-style type overrides
- **CoveragePoint / CoverageGroup**: Functional coverage tracking
- **RegisterModel / RegisterBlock**: Register Abstraction Layer with BitStruct

## File Structure

```
examples/
├── README.md                    # This file
├── 01_basic_runner/
├── 02_data_convert/
├── 03_bit_struct/
├── 04_driver_monitor/
├── 05_buffer/
├── 06_tile_mapping/
├── 07_uvm_basics/
└── 08_uvm_advanced/
```

## Viewing Waveforms

After running an example with a DUT, waveforms are available in the build directory:

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
- `torchbit/core/` - Vector and VectorSequence implementation
- `torchbit/tools/` - Driver, Monitor, Buffer classes
- `torchbit/tiling/` - TileMapping and AddressMapping
- `torchbit/uvm/` - UVM integration layer
- `torchbit/runner/` - Runner configuration
