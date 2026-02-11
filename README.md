<h1 align="center">Torchbit</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.5.0-blue" alt="version">
  <img src="https://img.shields.io/badge/Python-4109_lines-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Markdown-214_lines-083fa1" alt="Markdown">
  <img src="https://img.shields.io/badge/SystemVerilog-18_lines-dc382c" alt="SystemVerilog">
</p>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a>
</p>

Torchbit provides utilities for deep learning accelerator verification, facilitating the conversion of PyTorch tensors into Cocotb-compatible formats.

**Why Torchbit?** AI accelerator development should prioritize Python.

- All data rearrangement, tiling, and padding should be implemented using advanced tensor processing libraries like `einops` and `torch` in Python, rather than using `for` loops in Verilog or C++.
- Compatibility with `torch` is the top priority, ensuring seamless integration with the algorithmic Golden Model.

![logo](logo.jpg)

# Features

- **Rapidly build torch-native test frameworks:**
    - Map any `torch` data type to `cocotb` framework stimuli.
    - Abstract the Tensor mapping process based on `einops`.
    - Includes reusable components (e.g., Buffer, FIFO).
- **Fast debugging** (currently under development).

# Get Started

## Installation

```bash
pip install git+https://github.com/Devil-SX/torchbit.git
```

The main branch is currently maintained for environments using `cocotb >= 2.x`, Verilator >= 5.036, and VCS.



## Compatibility

| OS | Cocotb  | Simulator | Status | Notes |
|----|---------|-----------|--------|-------|
| WSL Ubuntu 22.04 | 2.0.0 | Verilator 5.038 | ✅ |  |
| WSL Ubuntu 22.04 | 2.0.0 | Verilator 5.036 | ❌ | fst assert failed, see [this issue](https://github.com/cocotb/cocotb/issues/4522) |
| CentOS 7 | 2.0.0 | VCS | ✅ |  |


# Philosophy

Tensor processing involves two key aspects: value processing and shape processing. Value processing deals with mapping various torch-supported data formats (float, signed, unsigned, brain-float) to Verilog bits. Shape processing includes complex Tilling transformations and other processes. For underlying value transformation principles, refer to [value.md](./doc/en/value.md). For tilling design principles, refer to [tilling_schedule.md](./doc/en/tilling_schedule.md).

# Basic Datatypes

Torchbit defines two sets of terminology for the software side (PyTorch) and the hardware side (Cocotb/HDL), connected by conversion methods.

## Terminology

| | PyTorch (Software) | Cocotb / HDL (Hardware) |
|---|---|---|
| **Single** | **Array** — 1D Tensor | **Logic** — packed integer for any signal |
| | | **Vector** — Array wrapper, SIMD parallel interface |
| | | **BitStruct** — field-defined interface |
| **Sequence** | **Matrix** — 2D Tensor | **LogicSequence** — sequence of Logic values |
| | | **VectorSequence** — sequence of Vectors |

## Transaction: Single Value Conversion

[Vector](./torchbit/core/vector.py) is the core bridge between PyTorch Arrays and HDL Logic values. `BitStruct` defines custom field layouts that also convert to/from Logic.

![Transaction](./doc/pic/01_transaction.png)

### Array → Logic (drive a signal)

```python
from torchbit.core import Vector, array_to_logic

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

# Step by step
vec = Vector.from_array(x)          # Array → Vector
dut.io_din.value = vec.to_logic()   # Vector → Logic

# One-liner shortcut
dut.io_din.value = array_to_logic(x)
```

### Logic → Array (read a signal)

```python
from torchbit.core import Vector, logic_to_array

# Step by step
vec = Vector.from_logic(dut.io_dout.value, 5, torch.float32)
x = vec.to_array()                   # Logic → Vector → Array

# One-liner shortcut
x = logic_to_array(dut.io_dout.value, 5, torch.float32)
```

> **Backward compatibility:** `from_tensor`/`to_tensor`, `from_cocotb`/`to_cocotb`, `to_int`, `tensor_to_cocotb`/`cocotb_to_tensor` are all preserved as aliases.

## Sequence: Batch Value Conversion

For sequences of values (e.g., multiple clock cycles), `VectorSequence` bridges PyTorch 2D Matrices and HDL LogicSequences.

![Sequence](./doc/pic/04_sequence.png)

```python
from torchbit.core import VectorSequence

matrix = torch.randn(256, 4)                     # 2D Matrix
vs = VectorSequence.from_matrix(matrix)           # Matrix → VectorSequence
logic_seq = vs.to_logic_sequence()                # VectorSequence → LogicSequence
restored = VectorSequence.from_logic_sequence(logic_seq, 4, torch.float32)
```

> **Backward compatibility:** `from_tensor`/`to_tensor`, `to_int_sequence`/`from_int_sequence` are preserved as aliases.

## Tensor ↔ LogicSequence via TileMapping

In real hardware, a high-dimensional Tensor (e.g., `c h w`) needs to be reshaped and serialized into a time-ordered sequence of packed values for transmission. `TileMapping` handles this end-to-end:

```
Tensor  ──rearrange──►  Matrix (2D)  ──pack rows──►  LogicSequence
 (c h w)     einops       (c, h*w)      Vector          [int, ...]
```

The conversion decomposes into two stages:

1. **Rearrange** (shape): `einops.rearrange` reshapes the Tensor into a 2D Matrix, where each row represents one clock cycle's data (spatial dimension) and the number of rows equals the number of clock cycles (temporal dimension).
2. **Pack** (value): Each row of the Matrix is packed into a single integer via `Vector`, producing a `LogicSequence`.

```python
from torchbit.tiling import TileMapping, array_to_logic_seq, logic_seq_to_array

mapping = TileMapping(
    dtype=torch.float32,
    sw_einops="c h w",
    hw_einops="c (h w)",
    hw_temp_dim={"c": 3},
    hw_spat_dim={"h": 32, "w": 32},
)

tensor = torch.randn(3, 32, 32)

# Tensor → LogicSequence (high-level shortcut)
seq = array_to_logic_seq(tensor, mapping)

# LogicSequence → Tensor
restored = logic_seq_to_array(seq, mapping)
assert torch.allclose(tensor, restored)
```

You can also use the low-level shortcuts without a TileMapping, operating directly on a 2D Matrix:

```python
from torchbit.tiling import matrix_to_logic_seq, logic_seq_to_matrix

matrix = torch.randn(256, 4)
seq = matrix_to_logic_seq(matrix)                            # Matrix → LogicSequence
restored = logic_seq_to_matrix(seq, 4, torch.float32)        # LogicSequence → Matrix
```

> **Backward compatibility:** `tensor_to_cocotb_seq`/`cocotb_seq_to_tensor` are preserved as aliases for `array_to_logic_seq`/`logic_seq_to_array`.

## Tools Overview

![Driver & Monitor](./doc/pic/02_driver_monitor.png)

- **Driver**: Feeds a `LogicSequence` into DUT via front-door (data + valid) signals.
- **PoolMonitor / FIFOMonitor**: Collects DUT output into a `LogicSequence` via front-door signals.

![Buffer](./doc/pic/03_buffer.png)

- **Buffer**: Memory model with both front-door (HDL) and back-door (software) access.
- **TileMapping**: Rearranges a Tensor into a Matrix, then packs into a `LogicSequence`.
- **AddressMapping**: Generates address sequences for back-door memory access.

# Running with Verilator/VCS Using Built-in Runner

`torchbit.runner` includes pre-built Cocotb launch wrappers that configure common simulator parameters. You can either write your testbench using standard Cocotb interfaces or leverage Torchbit's helper functions.

First, create your main test file (`top_test.py`) and a source file list. The file list can use paths relative to any location.

```python
# top_test.py
from torchbit.runner import (
    Runner,
    FileConfig,
    BuildConfig,
    DEFAULT_VCS_BUILD_CONFIG, # DEFAULT_VERILATOR_BUILD_CONFIG if use verilator
    read_filelist
)
import cocotb

# User Modifiable Variables
file_config_name = "my_design_name" # Name used to identify the build_dir, can be set to the top module name.
filelist_path = "path to filelist" # Path to your Verilog/SystemVerilog filelist.
filelist_base_path = "path relative to filelist" # Base path for relative paths in the filelist.
top_design_name = "top_module_name" # The name of your top-level design module.
# include_dirs = ["inc_dir1", "inc_dir2"] # Add include directories if you have any.
output_dir = "output_files" # The parent directory where output files should be generated.



TOP_FILE_CONFIG = FileConfig(
    name=file_config_name,
    sources=read_filelist(filelist_path, base_path=filelist_base_path),
    top_design=top_design_name,
    includes=include_dirs
)

TOP_RUNNER = Runner(
    file_config=TOP_FILE_CONFIG,
    build_config=DEFAULT_VCS_BUILD_CONFIG,
    current_dir=output_dir
)

@cocotb.test()
async def testbench():
    # writing your testbench here

if __name__ == "__main__":
    TOP_RUNNER.test("top_test")

```

Then, simply run `python top_test.py`. After execution, a `sim_xx` folder will be generated under `output_dir`. Inside this directory, you will find the compiled files and corresponding waveforms. If using Verilator, the waveform file is `dump.fst`; if using VCS, it is `dump.fsdb`.

The `.fsdb` file stores the runtime database. You can view the corresponding source code directly by running `verdi -ssf dump.fsdb`.



