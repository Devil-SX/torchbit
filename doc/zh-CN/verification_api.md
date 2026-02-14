# 验证 API 参考

Torchbit 提供两种硬件验证 API 模式：

- **原生 torchbit** — 直接 cocotb 集成；轻量级，无额外依赖。
- **PyUVM 兼容** — 工厂函数生成 `pyuvm` 组件类；遵循 UVM 方法学。

两者共享相同的底层数据类型（`Vector`、`LogicSequence`、`TileMapping` 等），可在同一项目中混合使用。

## 目录

- [API 对照表](#api-对照表)
- [原生 Torchbit API](#原生-torchbit-api)
  - [信号端口](#信号端口)
  - [Driver](#driver)
  - [FIFODriver](#fifodriver)
  - [PoolMonitor](#poolmonitor)
  - [FIFOMonitor](#fifomonitor)
  - [FIFOReceiver](#fiforeceiver)
  - [Buffer / TwoPortBuffer](#buffer--twoportbuffer)
  - [TransferStrategy](#transferstrategy)
  - [ComponentDB](#componentdb)
  - [原生 API 工作流示例](#原生-api-工作流示例)
- [PyUVM 兼容 API](#pyuvm-兼容-api)
  - [TorchbitBFM](#torchbitbfm)
  - [序列项](#序列项)
  - [TorchbitScoreboard](#torchbitscoreboard)
  - [工厂函数](#工厂函数)
  - [ComponentRegistry](#componentregistry)
  - [功能覆盖率](#功能覆盖率)
  - [寄存器抽象层 (RAL)](#寄存器抽象层-ral)
  - [PyUVM 工作流示例](#pyuvm-工作流示例)
- [两种 API 的选择](#两种-api-的选择)

---

## API 对照表

| 验证概念 | 原生 Torchbit (`torchbit.tools`) | PyUVM 兼容 (`torchbit.uvm`) | 需要 pyuvm |
|---|---|---|:---:|
| **信号抽象** | `InputPort` / `OutputPort` | `TorchbitBFM` | 否 |
| **驱动激励** | `Driver` / `FIFODriver` | `create_uvm_driver()` | 是 |
| **采集响应** | `PoolMonitor` / `FIFOMonitor` / `FIFOReceiver` | `create_uvm_monitor()` | 是 |
| **结果比对** | 手动 `torch.equal()` | `TorchbitScoreboard` | 否 |
| **序列项** | `LogicSequence`（普通列表） | `VectorItem` / `LogicSequenceItem` | 否 |
| **Agent** | *（无对应——手动组合）* | `create_uvm_agent()` | 是 |
| **环境** | *（无对应）* | `TorchbitEnv` / `create_uvm_env()` | 否 / 是 |
| **测试** | `@cocotb.test()` 函数 | `TorchbitTest` / `create_uvm_test()` | 否 / 是 |
| **传输时序** | `TransferStrategy` 子类 | 同（`TransferStrategy`） | 否 |
| **存储模型** | `Buffer` / `TwoPortBuffer` | 同 | 否 |
| **组件发现** | `ComponentDB`（基于路径） | `ComponentRegistry`（类型覆盖） | 否 |
| **功能覆盖率** | *（无对应）* | `CoverageGroup` / `CoveragePoint` | 否 |
| **寄存器模型** | *（无对应）* | `RegisterModel` / `RegisterBlock` | 否 |

---

## 原生 Torchbit API

所有类位于 `torchbit.tools`，直接与 cocotb 配合使用——无需 pyuvm 依赖。

### 信号端口

```python
from torchbit.tools import InputPort, OutputPort
```

| 类 | 用途 | 关键方法 |
|---|------|---------|
| `InputPort(signal)` | 读取 DUT 信号 | `get() -> int` |
| `OutputPort(signal, set_immediately=False)` | 驱动 DUT 信号 | `set(value: int)` |

两者均能优雅处理 `None`（返回 0 / 空操作）。

### Driver

```python
from torchbit.tools import Driver
```

原始激励驱动器。通过 `data` + `valid` 信号发送 `LogicSequence`，可选 `full` 反压。

```python
driver = Driver(debug=False)
driver.connect(dut, clk, data=dut.din, valid=dut.din_vld, full=None)
driver.load([0x1, 0x2, 0x3])
await driver.run()
```

| 方法 | 签名 |
|------|------|
| `connect` | `(dut, clk, data, valid, full=None)` |
| `load` | `(sequence: LogicSequence)` |
| `run` | `async (stop_event=None)` |
| `dump_time` | `() -> List[tuple]` |

### FIFODriver

```python
from torchbit.tools import FIFODriver
```

增强版驱动器，支持可插拔的 `TransferStrategy` 和显式极性控制（`active_high`）。

```python
driver = FIFODriver(strategy=RandomBackpressure(0.3))
driver.connect(dut, clk, data=dut.din, valid=dut.din_vld, ready=dut.din_rdy)
driver.load([0x1, 0x2, 0x3])
await driver.run()
```

支持 `ComponentDB` 基于路径创建：

```python
driver = FIFODriver.from_path("top.encoder.input", dut, dut.clk)
```

### PoolMonitor

```python
from torchbit.tools import PoolMonitor
```

始终就绪的接收器。当 `valid` 有效时采集数据。

```python
monitor = PoolMonitor(debug=False)
monitor.connect(dut, clk, data=dut.dout, valid=dut.dout_vld)

stop_event = Event()
cocotb.start_soon(monitor.run(stop_event))
# ... 运行仿真 ...
stop_event.set()
results = monitor.dump()          # LogicSequence
```

### FIFOMonitor

```python
from torchbit.tools import FIFOMonitor
```

带 FIFO 流控的接收器。根据 `empty` 信号驱动 `ready`。

```python
monitor = FIFOMonitor(debug=False)
monitor.connect(dut, clk, data=dut.dout, empty=dut.fifo_empty,
                ready=dut.fifo_ready, valid=dut.dout_vld)
```

### FIFOReceiver

```python
from torchbit.tools import FIFOReceiver
```

增强版接收器，基于 `TransferStrategy` 控制 `ready` 信号。在新设计中替代 `FIFOMonitor`。

```python
receiver = FIFOReceiver(strategy=BurstStrategy(burst_len=8, pause_cycles=4))
receiver.connect(dut, clk, data=dut.dout, valid=dut.dout_vld, ready=dut.dout_rdy)
```

### Buffer / TwoPortBuffer

```python
from torchbit.tools import Buffer, TwoPortBuffer
```

内存模型，支持后门访问。`TwoPortBuffer` 增加了双端口 HDL 连接。

```python
buf = TwoPortBuffer(width=32, depth=256, backpressure=False)
buf.connect(dut, clk,
            wr_csb=dut.wr_csb, wr_din=dut.wr_din, wr_addr=dut.wr_addr,
            rd_csb=dut.rd_csb, rd_addr=dut.rd_addr,
            rd_dout=dut.rd_dout, rd_dout_vld=dut.rd_dout_vld)
await buf.init()
cocotb.start_soon(buf.run())

# 后门访问（即时，无需时钟）
buf.write(0x00, 0xDEADBEEF)
val = buf.read(0x00)

# 张量后门（需要 TileMapping + AddressMapping）
buf.backdoor_load_tensor(tensor, tile_mapping, addr_mapping)
result = buf.backdoor_dump_tensor(tile_mapping, addr_mapping)
```

### TransferStrategy

```python
from torchbit.tools import (
    GreedyStrategy, RandomBackpressure, BurstStrategy, ThrottledStrategy,
)
```

所有策略实现 `should_transfer(cycle: int) -> bool`：

| 策略 | 构造函数 | 行为 |
|------|---------|------|
| `GreedyStrategy` | `()` | 始终传输 |
| `RandomBackpressure` | `(stall_prob=0.3, seed=None)` | 按概率暂停 |
| `BurstStrategy` | `(burst_len=8, pause_cycles=4)` | N 周期开、M 周期停模式 |
| `ThrottledStrategy` | `(min_interval=3)` | 每 N 周期传输一次 |

### ComponentDB

```python
from torchbit.tools import ComponentDB
```

层次化路径信号注册表。注册一次，随处解析。

```python
ComponentDB.set("top.encoder.input", {
    "data": dut.din, "valid": dut.din_vld, "ready": dut.din_rdy,
})
driver = FIFODriver.from_path("top.encoder.input", dut, dut.clk)
```

### 原生 API 工作流示例

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Event
from torchbit.tools import Driver, PoolMonitor

@cocotb.test()
async def test_pipe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # 驱动
    driver = Driver(debug=True)
    driver.connect(dut, dut.clk, dut.din, dut.din_vld, full=None)
    driver.load([10, 20, 30])

    # 监控
    monitor = PoolMonitor()
    monitor.connect(dut, dut.clk, dut.dout, dut.dout_vld)
    stop_event = Event()
    cocotb.start_soon(monitor.run(stop_event))

    await driver.run()

    for _ in range(10):
        await RisingEdge(dut.clk)
    stop_event.set()

    assert list(monitor.dump()) == [10, 20, 30]
```

---

## PyUVM 兼容 API

组件位于 `torchbit.uvm`。**基础类**（`TorchbitBFM`、`TorchbitScoreboard`、`VectorItem` 等）作为纯 Python 运行——无需 pyuvm。**工厂函数**（`create_uvm_driver`、`create_uvm_agent` 等）生成 `pyuvm` 子类，需要 `pyuvm >= 4.0.0`。

### TorchbitBFM

```python
from torchbit.uvm import TorchbitBFM
```

信号抽象层——按名称注册 DUT 信号，通过统一接口驱动/采样。

```python
bfm = TorchbitBFM(dut=dut, clk=dut.clk)
bfm.add_input('din', dut.din)           # TB 驱动此信号
bfm.add_input('din_vld', dut.din_vld)
bfm.add_output('dout', dut.dout)         # TB 采样此信号
bfm.add_output('dout_vld', dut.dout_vld)

bfm.drive('din', 0xBEEF)
bfm.drive('din_vld', 1)
val = bfm.sample('dout')
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `add_input` | `(name, signal)` | 注册 DUT 输入（封装为 `OutputPort`） |
| `add_output` | `(name, signal)` | 注册 DUT 输出（封装为 `InputPort`） |
| `drive` | `(name, value)` | 驱动已注册的输入 |
| `sample` | `(name) -> int` | 采样已注册的输出 |
| `has_input` | `(name) -> bool` | 检查注册状态 |
| `has_output` | `(name) -> bool` | 检查注册状态 |

### 序列项

```python
from torchbit.uvm import VectorItem, LogicSequenceItem
```

为 torchbit 数据类型添加 UVM 序列项语义（命名、可比较）的封装。

```python
from torchbit.core import Vector
import torch

item = VectorItem("stim_0", Vector.from_array(torch.tensor([1, 2, 3])))
other = VectorItem("stim_1", Vector.from_array(torch.tensor([1, 2, 3])))
assert item == other   # 通过 torch.equal() 比较底层张量
```

### TorchbitScoreboard

```python
from torchbit.uvm import TorchbitScoreboard
```

期望值与实际值的比对引擎。作为纯 Python 运行。

```python
sb = TorchbitScoreboard(name="my_scoreboard")

sb.add_expected(VectorItem("exp", vec_a))
sb.add_actual(VectorItem("act", vec_b))      # 立即触发比较

print(sb.match_count, sb.mismatch_count)
assert sb.passed
print(sb.report())
```

| 方法 / 属性 | 说明 |
|---|---|
| `add_expected(item)` | 添加期望项 |
| `add_actual(item)` | 添加实际项；如有期望项则立即比较 |
| `passed: bool` | 无失配且所有队列已清空时为 `True` |
| `match_count: int` | 匹配次数 |
| `mismatch_count: int` | 失配次数 |
| `report() -> str` | 格式化摘要 |

### 工厂函数

生成 `pyuvm` 子类，需要 `pyuvm >= 4.0.0`。

```python
from torchbit.uvm.driver import create_uvm_driver
from torchbit.uvm.monitor import create_uvm_monitor
from torchbit.uvm.scoreboard import create_uvm_scoreboard
from torchbit.uvm.agent import create_uvm_agent
from torchbit.uvm.env import create_uvm_env
from torchbit.uvm.test import create_uvm_test
```

| 工厂函数 | 返回 | 基类 | 关键 phase |
|---------|------|------|-----------|
| `create_uvm_driver(name)` | `uvm_driver` 子类 | `pyuvm.uvm_driver` | `build_phase`（BFM + STRATEGY 从 ConfigDB 获取），`run_phase`（驱动循环） |
| `create_uvm_monitor(name)` | `uvm_monitor` 子类 | `pyuvm.uvm_monitor` | `build_phase`（analysis port + BFM），`run_phase`（采样循环） |
| `create_uvm_scoreboard(name)` | `uvm_component` 子类 | `pyuvm.uvm_component` | `build_phase`（FIFO），`check_phase`（清空 + 断言） |
| `create_uvm_agent(driver_cls, monitor_cls, name)` | `uvm_agent` 子类 | `pyuvm.uvm_agent` | `build_phase`（sequencer + driver + monitor），`connect_phase` |
| `create_uvm_env(agent_configs, name)` | `uvm_env` 子类 | `pyuvm.uvm_env` | `build_phase`（agents + scoreboard），`connect_phase` |
| `create_uvm_test(env_cls, name)` | `uvm_test` 子类 | `pyuvm.uvm_test` | `build_phase`（env） |

### ComponentRegistry

```python
from torchbit.uvm import ComponentRegistry
```

类型覆盖注册表（UVM 工厂模式）。

```python
ComponentRegistry.set_override("Driver", MyCustomDriver)
cls = ComponentRegistry.get("Driver", default=DefaultDriver)
```

### 功能覆盖率

```python
from torchbit.uvm import CoverageGroup, CoveragePoint
```

带分箱的功能覆盖率跟踪。

```python
cg = CoverageGroup("transfer_cov")
cg.add_point("size", [
    ("small",  lambda v: v < 16),
    ("medium", lambda v: 16 <= v < 64),
    ("large",  lambda v: v >= 64),
])
cg.sample("size", 32)
print(cg.coverage_pct, cg.report())
```

### 寄存器抽象层 (RAL)

```python
from torchbit.uvm import RegisterModel, RegisterBlock
```

基于 `BitStruct` 定义的寄存器建模，支持通过 `Buffer` 进行后门访问。

```python
block = RegisterBlock("ctrl", base_addr=0x1000)
reg = block.add_register("config", ConfigBitStruct, offset=0)
block.write("config", "enable", 1)
val = block.read("config", "enable")

# 通过 Buffer 后门访问
block.backdoor_write(buffer, "config")
block.backdoor_read(buffer, "config")
```

### PyUVM 工作流示例

使用 BFM + Scoreboard 配合 cocotb（无需 pyuvm 依赖）：

```python
import cocotb
import torch
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Event
from torchbit.core import Vector
from torchbit.uvm import TorchbitBFM, TorchbitScoreboard, VectorItem
from torchbit.tools.strategy import GreedyStrategy

@cocotb.test()
async def test_pipe_uvm(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # BFM：注册信号
    bfm = TorchbitBFM(dut=dut, clk=dut.clk)
    bfm.add_input('din', dut.din)
    bfm.add_input('din_vld', dut.din_vld)
    bfm.add_output('dout', dut.dout)
    bfm.add_output('dout_vld', dut.dout_vld)

    # Scoreboard：加载期望值
    sb = TorchbitScoreboard(name="sb")
    sequence = [10, 20, 30]
    for v in sequence:
        sb.add_expected(VectorItem(f"e_{v}",
            Vector.from_array(torch.tensor([v], dtype=torch.int32))))

    # 监控任务
    stop_event = Event()
    async def monitor():
        while not stop_event.is_set():
            await RisingEdge(dut.clk)
            if int(bfm.sample('dout_vld')) == 1:
                val = int(bfm.sample('dout'))
                sb.add_actual(VectorItem(f"a_{val}",
                    Vector.from_array(torch.tensor([val], dtype=torch.int32))))
    cocotb.start_soon(monitor())

    # 通过 BFM 驱动
    for v in sequence:
        bfm.drive('din', v)
        bfm.drive('din_vld', 1)
        await RisingEdge(dut.clk)
    bfm.drive('din_vld', 0)

    await ClockCycles(dut.clk, 10)
    stop_event.set()

    assert sb.passed, sb.report()
```

完整 pyuvm 工厂工作流（需要 `pyuvm >= 4.0.0`）：

```python
from torchbit.uvm.driver import create_uvm_driver
from torchbit.uvm.monitor import create_uvm_monitor
from torchbit.uvm.agent import create_uvm_agent
from torchbit.uvm.env import create_uvm_env
from torchbit.uvm.test import create_uvm_test

DriverCls = create_uvm_driver()
MonitorCls = create_uvm_monitor()
AgentCls = create_uvm_agent(DriverCls, MonitorCls)
EnvCls = create_uvm_env(agent_configs=[{"name": "pipe_agent"}])
TestCls = create_uvm_test(env_cls=EnvCls)
```

---

## 两种 API 的选择

| 考量因素 | 原生 torchbit | PyUVM 兼容 |
|---|---|---|
| **依赖** | 仅 cocotb | cocotb + pyuvm |
| **学习曲线** | 较低——普通异步函数 | 较高——UVM phase、ConfigDB、sequencer |
| **可复用性** | 手动组合 | 系统化（agent/env/test 层次结构） |
| **Scoreboard** | 手动 `torch.equal` | 内置 `TorchbitScoreboard` |
| **覆盖率** | 不可用 | `CoverageGroup` / `CoveragePoint` |
| **寄存器模型** | 不可用 | `RegisterModel` / `RegisterBlock` |
| **适用场景** | 小模块、快速原型 | 大型设计、多 agent 环境 |

**推荐混合使用。** 典型的折中方案：

- 使用 `TorchbitBFM` 进行信号抽象（无需 pyuvm）。
- 使用 `TorchbitScoreboard` + `VectorItem` 进行结果比对（无需 pyuvm）。
- 使用原生 `TwoPortBuffer` 进行存储建模。
- 使用 `TransferStrategy` 进行时序控制。
- 除非需要完整的 UVM phase 机制，否则跳过 pyuvm 工厂函数。

参见 `examples/09_uvm_pipeline/` 获取此混合方案的实际示例。
