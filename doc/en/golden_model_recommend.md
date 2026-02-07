This document discusses recommended practices for Golden Models in AI accelerator verification.

As mentioned in [tilling_schedule.md](../zh-CN/tilling_schedule.md), software consists of two parts: logic and scheduling. High-level software languages like PyTorch only contain the logic part. Therefore, our strategy is:

- Reuse the PyTorch ecosystem; Golden Model code is compatible with PyTorch
- Supplement scheduling information on top of PyTorch
- Support PyTorch algorithm composition for end-to-end testing

## Example: Addition Operation

Taking addition as an example, here is the algorithm prototype:

```python
def add(a, b):
    return a + b
```

Here is the corresponding hardware interface:

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

## Information Comparison

### Information Contained in Algorithm Prototype

- Specific values of data `a` and `b`
- Logical operation (addition)

### Information Missing from Algorithm Prototype

- Hardware scheduling tiling strategy: how much data to compute per time step, how many blocks to split into
- Buffer interaction address information
- Unit instruction information: data dimensions, operation types, etc.

## torchbit Solution

torchbit's `TileMapping` abstraction supports mapping between [Software-style Tensor](../zh-CN/tilling_schedule.md) and Hardware-style Matrix, while instruction information can be described and defined using `BitStruct`.

### Data Loading Paths

- **Path 1**: Software Tensor &rarr; TileMapping &rarr; Hardware Matrix &rarr; Backdoor Write &rarr; Driver (FIFO/Buffer) &rarr; Front-door Interaction &rarr; Hardware Unit
- **Path 2**: Software Tensor &rarr; Shape Dimension Parsing &rarr; Instruction Generation &rarr; Hardware Unit

### Data Saving Path

Hardware Unit &rarr; Front-door Interaction &rarr; Receiver (FIFO/Buffer) &rarr; Backdoor Read &rarr; Hardware Matrix &rarr; TileMapping Reverse Mapping &rarr; Software Tensor

## Pure Software Golden Model

Since `TileMapping` supports bidirectional mapping, even with a pure software Golden Model (without hardware information like addresses and instructions), data result correctness can still be compared directly at the Software Tensor level.
