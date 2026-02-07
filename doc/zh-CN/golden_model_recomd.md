本文讨论 AI 加速器验证中 Golden Model 的推荐实践。

正如 [tilling_schedule.md](./tilling_schedule.md) 所述，软件分为逻辑和调度两部分，而高层软件语言（如 PyTorch）只包含逻辑部分。因此我们的策略是：

- 复用 PyTorch 生态，Golden Model 写法兼容 PyTorch
- 在 PyTorch 基础上补充调度信息
- 支持 PyTorch 算法拼接运行端到端测试

## 示例：加法运算

以加法为例，这是算法原型：

```python
def add(a, b):
    return a + b
```

这是硬件对应的接口：

```verilog
module add_unit (
    input logic clk,
    input logic rst_n,

    input logic [INST_WIDTH-1:0] inst,
    input logic inst_valid,
    input logic start,
    output logic done,

    output logic [ADDR_WIDTH-1:0] a_addr,
    output logic a_addr_valid,
    output logic [ADDR_WIDTH-1:0] b_addr,
    output logic b_addr_valid,
    input logic [DATA_WIDTH-1:0] a_data,
    input logic a_valid,
    input logic [DATA_WIDTH-1:0] b_data,
    input logic b_valid,

    output logic [DATA_WIDTH-1:0] result_data,
    output logic [ADDR_WIDTH-1:0] result_addr,
    output logic result_valid
);
```

## 信息对比

### 算法原型包含的信息

- 数据 `a` 和 `b` 的具体数值
- 逻辑操作（加法）

### 算法原型缺少的信息

- 硬件调度的 tiling 策略：单位时间步计算多少数据、切分成多少块
- Buffer 交互的地址信息
- 单元的指令信息：数据维度、操作类型等

## torchbit 解决方案

torchbit 的 `TileMapping` 抽象支持在 [Software-style Tensor](./tilling_schedule.md) 和 Hardware-style Matrix 之间进行映射，而指令信息支持使用 `BitStruct` 进行描述定义。

### 数据加载路径

- **路径1**：软件 Tensor &rarr; TileMapping &rarr; Hardware Matrix &rarr; 后门写入 &rarr; Driver (FIFO/Buffer) &rarr; 前门交互 &rarr; Hardware Unit
- **路径2**：软件 Tensor &rarr; shape 维度解析 &rarr; 指令信息生成 &rarr; Hardware Unit

### 数据保存路径

Hardware Unit &rarr; 前门交互 &rarr; Receiver (FIFO/Buffer) &rarr; 后门读出 &rarr; Hardware Matrix &rarr; TileMapping 逆映射 &rarr; 软件 Tensor

## 纯软件 Golden Model

由于 `TileMapping` 支持双向映射，即使使用纯软件 Golden Model，虽然没有地址、指令等硬件信息，也可以直接在 Software Tensor 层面比较数据结果的正确性。
