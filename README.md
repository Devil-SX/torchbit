<h1 align="center">Torchbit</h1>

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
git clone https://github.com/Devil-SX/torchbit.git
cd torchbit
pip install -e .
```

The main branch is currently maintained for environments using `cocotb >= 2.x`, Verilator >= 5.036, and VCS.

## Development



## Compatibility

| OS | Cocotb  | Simulator | Status | Notes |
|----|---------|-----------|--------|-------|
| WSL Ubuntu 22.04 | 2.0.0 | Verilator 5.038 | ✅ |  |
| WSL Ubuntu 22.04 | 2.0.0 | Verilator 5.036 | ❌ | fst assert failed, see [this issue](https://github.com/cocotb/cocotb/issues/4522) |
| CentOS 7 | 2.0.0 | VCS | ✅ |  |


## Basic Concept

[Vector](./torchbit/core/vector.py) is a specialized 1D Tensor and serves as the fundamental data type. The `Vector` class acts as the interface between PyTorch Tensors and Cocotb LogicArrays or Verilog multi-bit interfaces.

### Tensor -> Cocotb

For example, to convert a Tensor `x` (length 5, `torch.float32`) into a Verilog signal—resulting in a 5x32=160-bit signal—and drive it into a 160-bit wide interface `dut.io_din`:

**First Step: Convert Tensor to Vector**

```python
from torchbit.core import Vector
x_vec = Vector.from_tensor(x) # x is a 1D-tensor with 5 torch.float32 elements
```

**Second Step: Convert Vector to Cocotb format and drive the signal**

```python
dut.io_din.value = x_vec.to_cocotb()
```

Alternatively, this process can be combined into a single step:

```python
dut.io_din.value = Vector.from_tensor(x).to_cocotb()

from torchbit.core import tensor_to_cocotb
dut.io_din.value = tensor_to_cocotb(x) # wrapper of Vector.from_tensor().to_cocotb()
```

### Cocotb -> Tensor

Similarly, to read a Tensor `x` (length 5, `torch.float32`) from a 160-bit wide interface `dut.io_dout`:

```python
x = Vector.from_cocotb(dut.io_dout.value, 5, torch.float32).to_tensor()

from torchbit.core import cocotb_to_tensor
x = cocotb_to_tensor(dut.io_dout.value, 5, torch.float32) # wrapper of Vector.from_cocotb().to_tensor()
```

This tutorial covers the essential Tensor conversion interfaces. You can write your testbench entirely in the [cocotb](https://docs.cocotb.org/en/stable/writing_testbenches.html) style, using this library purely as a value conversion tool.

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


# Advanced Usage


## Tiling / Padding

> todo

## IP

> todo

- FIFO
- Buffer

## Debug Tools

> todo