<h1 align="center">Torchbit</h1>

[English](./README.md) | [简体中文](./README.zh-CN.md)

Torchbit 为深度学习加速器验证提供实用工具，便于将 PyTorch 张量转换为 Cocotb 兼容格式。

**为什么选择 Torchbit？** AI 加速器开发应优先考虑 Python。

- 所有数据重排、分块和填充都应使用 `einops` 和 `torch` 等高级张量处理库在 Python 中实现，而不是在 Verilog 或 C++ 中使用 `for` 循环。
- 与 `torch` 的兼容性是首要任务，确保与算法 Golden Model 无缝集成。

![logo](logo.jpg)

# 功能特性

- **快速构建 torch 原生测试框架：**
    - 将任意 `torch` 数据类型映射到 `cocotb` 框架激励。
    - 基于 `einops` 抽象 Tensor 映射过程。
    - 包含可重用组件（如 Buffer、FIFO）。
- **快速调试**（目前开发中）。

# 快速开始

## 安装

```bash
git clone https://github.com/Devil-SX/torchbit.git
cd torchbit
pip install -e .
```

主分支目前为使用 `cocotb >= 2.x`、Verilator >= 5.036 和 VCS 的环境维护。



## 兼容性

| 操作系统 | Cocotb  | 模拟器 | 状态 | 备注 |
|----|---------|-----------|--------|-------|
| WSL Ubuntu 22.04 | 2.0.0 | Verilator 5.038 | ✅ |  |
| WSL Ubuntu 22.04 | 2.0.0 | Verilator 5.036 | ❌ | fst 断言失败，参见 [此 issue](https://github.com/cocotb/cocotb/issues/4522) |
| CentOS 7 | 2.0.0 | VCS | ✅ |  |


# 哲学

Tensor 的处理主要分为两个关键，对值的处理，以及对形状的处理。对值的处理分为如何将各种 torch 支持的数据格式 float、signed、unsigned、brain-float 映射到 Verilog 的 bits 上。对形状的处理则包括复杂的 Tilling 变换等过程。底层值变换原理参考 [value.md](./doc/zh-CN/value.md)，tilling 设计原理参考 [tilling_schedule.md](./doc/zh-CN/tilling_schedule.md)。

# 基本数据结构

[Vector](./torchbit/core/vector.py) 是一个专门的 1D Tensor，作为基本数据类型。`Vector` 类充当 PyTorch Tensor 与 Cocotb LogicArrays 或 Verilog 多位接口之间的接口。

## Tensor -> Cocotb

例如，要将 Tensor `x`（长度 5，`torch.float32`）转换为 Verilog 信号——结果是 5x32=160 位信号——并将其驱动到 160 位宽的接口 `dut.io_din`：

**第一步：将 Tensor 转换为 Vector**

```python
from torchbit.core import Vector
x_vec = Vector.from_tensor(x) # x 是一个包含 5 个 torch.float32 元素的 1D tensor
```

**第二步：将 Vector 转换为 Cocotb 格式并驱动信号**

```python
dut.io_din.value = x_vec.to_cocotb()
```

或者，这两步可以合并为一步：

```python
dut.io_din.value = Vector.from_tensor(x).to_cocotb()

from torchbit.core import tensor_to_cocotb
dut.io_din.value = tensor_to_cocotb(x) # Vector.from_tensor().to_cocotb() 的封装
```

## Cocotb -> Tensor

同样，要从 160 位宽的接口 `dut.io_dout` 读取 Tensor `x`（长度 5，`torch.float32`）：

```python
x = Vector.from_cocotb(dut.io_dout.value, 5, torch.float32).to_tensor()

from torchbit.core import cocotb_to_tensor
x = cocotb_to_tensor(dut.io_dout.value, 5, torch.float32) # Vector.from_cocotb().to_tensor() 的封装
```

本教程涵盖了基本的 Tensor 转换接口。您可以完全按照 [cocotb](https://docs.cocotb.org/en/stable/writing_testbenches.html) 风格编写测试平台，将此库纯粹用作值转换工具。

# 使用内置运行器在 Verilator/VCS 上运行

`torchbit.runner` 包括预构建的 Cocotb 启动封装，可配置常见模拟器参数。您可以使用标准 Cocotb 接口编写测试平台，也可以利用 Torchbit 的辅助函数。

首先，创建主测试文件（`top_test.py`）和源文件列表。文件列表可以使用相对于任何位置的路径。

```python
# top_test.py
from torchbit.runner import (
    Runner,
    FileConfig,
    BuildConfig,
    DEFAULT_VCS_BUILD_CONFIG, # 如果使用 verilator，使用 DEFAULT_VERILATOR_BUILD_CONFIG
    read_filelist
)
import cocotb

# 用户可修改变量
file_config_name = "my_design_name" # 用于标识 build_dir 的名称，可以设置为顶层模块名称。
filelist_path = "filelist 路径" # Verilog/SystemVerilog filelist 的路径。
filelist_base_path = "相对于 filelist 的路径" # filelist 中相对路径的基础路径。
top_design_name = "top_module_name" # 顶层设计模块的名称。
# include_dirs = ["inc_dir1", "inc_dir2"] # 如果有包含目录，请添加。
output_dir = "output_files" # 生成输出文件的父目录。



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
    # 在此处编写测试平台

if __name__ == "__main__":
    TOP_RUNNER.test("top_test")

```

然后，只需运行 `python top_test.py`。执行后，将在 `output_dir` 下生成一个 `sim_xx` 文件夹。在此目录中，您可以找到编译文件和相应的波形。如果使用 Verilator，波形文件是 `dump.fst`；如果使用 VCS，则是 `dump.fsdb`。

`.fsdb` 文件存储运行时数据库。您可以通过运行 `verdi -ssf dump.fsdb` 直接查看相应的源代码。



