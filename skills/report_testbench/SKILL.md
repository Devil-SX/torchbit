---
name: report_testbench
description: "Generate a testbench architecture report for a hardware module. Use when you need to understand or document a module's verification environment. Triggers on: report testbench, describe testbench, testbench report, explain testbench, testbench architecture."
user-invocable: true
argument-hint: <module_name>
---

# Testbench Architecture Report

Generate a clear, visual report that explains how a hardware module's testbench is constructed.

The argument `<module_name>` refers to a DUT module directory under `tb/` (e.g. `dut_buffer`, `dut_tile_mover`). Read the wrapper, tests, and golden model files for that module before writing the report.

---

## What Makes a Good Report

A good testbench report is one that a new team member can read and immediately understand **what the DUT does, how its interfaces are exercised, and where the test data comes from** — without needing to read every line of source code. Below are the four pillars of a good report.

---

### 1. ASCII Art Interface Diagram

The report should open with a character-art block diagram showing the DUT and its ports. This gives the reader a spatial intuition of the module before any text explanation.

Good diagrams:
- Show the DUT as a box with labeled ports on each side (inputs left, outputs right, or grouped logically)
- Group related signals together (e.g. `wr_addr`, `wr_din`, `wr_csb` as a write-port bundle)
- Show which testbench component connects to each port group (Driver, Monitor, TwoPortBuffer, etc.)
- Include bit-widths on key signals

Example style (not prescriptive — adapt to the DUT):

```
                         +-----------------+
   clk ────────────────> |                 |
   rst_n ──────────────> |                 |
                         |                 |
   ctrl_start ─────────> |   MemoryMover   | ───> done
   ctrl_src_base[15:0] > |                 |
   ctrl_dst_base[15:0] > |                 |
                         |                 |
   wr_addr[7:0]  <────  |                 |  ────> rd_addr[7:0]
   wr_din[31:0]  <────  |                 |  ────> rd_dout[31:0]
   wr_csb        <────  |                 |  ────> rd_csb
                         +-----------------+
```

The diagram should make the grouping of control, data-write, and data-read interfaces immediately visible.

---

### 2. Testbench Topology Diagram

In addition to the DUT port diagram, include a **component-level topology diagram** showing how testbench components (Test, Wrapper, TwoPortBuffer, Driver, Monitor, Golden Model, etc.) connect to each other and to the DUT.

This diagram answers: "who drives whom, and how does data flow through the verification environment?"

Good topology diagrams:
- Show every torchbit component as a box
- Label each connection with the access method (e.g. `backdoor_load_tensor()`, `connect()`, `run()`)
- Distinguish frontdoor access (through DUT signals, cycle-accurate) from backdoor access (direct Python memory read/write, instant)
- Show the golden model's position: where its inputs come from and where its output is compared

Example style (adapt to the DUT):

```
  ┌──────────┐   backdoor write   ┌──────────────┐    frontdoor     ┌─────────┐
  │   Test   │ ──load_source───> │ TwoPortBuffer │ <──rd/wr port──> │   DUT   │
  │ (cocotb) │ <──read_dest───── │  (w=32,d=256) │                  │ MemMover│
  └────┬─────┘                   └──────────────┘                  └─────────┘
       │                                                               │
       │  load_data()    ┌──────────────┐    compare (torch.equal)     │
       └────────────────>│ Golden Model │<──────── dut_result ─────────┘
                         └──────────────┘
```

This diagram is often more informative than the port diagram for understanding the verification strategy.

---

### 3. Stimulus Framework Description

For each group of interfaces, explain **what torchbit component drives or monitors it** and **what method it uses**.

Good descriptions:
- Use torchbit class names precisely: `Driver`, `PoolMonitor`, `TwoPortBuffer`, `Vector`, `VectorSequence`, `LogicSequence`, `TileMapping`, `AddressMapping`, `ContiguousAddressMapping`
- Explain the connection between the testbench component and the DUT signals — e.g. which signals are passed to `Driver.connect()` or `TwoPortBuffer.connect()`
- Describe the data flow direction and timing behavior (handshake protocol, backdoor vs. frontdoor, blocking vs. fire-and-forget)
- If a golden model exists, explain what it computes and how its output is compared against DUT results

For instance, rather than saying "data is loaded into memory", a good report says:
> The source tensor is loaded via `TwoPortBuffer.backdoor_load_tensor()`, which uses both a `TileMapping` (to convert the tensor layout) and an `AddressMapping` (to generate write addresses). The monitor side reads back results via `TwoPortBuffer.dump_to_tensor()` with the destination address mapping.

---

### 4. Test Data Description

The most important section for data-intensive accelerator testbenches. A good report explains:

- **What the test data is**: its semantic meaning (e.g. "a batch of feature maps", "weight matrix")
- **Tensor dimensions**: describe using dimension names and sizes, e.g. `(batch=2, width=4, channel=8)` — not just `(2, 4, 8)`
- **dtype**: the torch dtype used (e.g. `torch.int16`, `torch.float32`)
- **How data is imported**: is it `torch.arange()`, random (`torch.randn()`), loaded from file (`VectorSequence.from_memhexfile()`), or computed by a golden model?
- If there are multiple data regions (source, destination, weights), describe each separately

#### Data Layout Abstractions: TileMapping & AddressMapping

The report **must explicitly state** whether the testbench uses `TileMapping` and/or `AddressMapping` abstractions. This is a key architectural decision that distinguishes simple testbenches from those with complex tensor-to-hardware layout transformations.

- **If neither is used**: state this clearly (e.g. "This testbench does not use TileMapping or AddressMapping. Data is loaded via direct per-address backdoor writes using `buffer.write(addr, value)`."). Then describe the actual data loading mechanism.

- **If TileMapping is used**: show the full mapping with all parameters:
  - `sw_einops` — the software-side einops expression
  - `hw_einops` — the hardware-side einops expression, clearly marking which dims are temporal vs. spatial
  - `hw_temp_dim` — temporal dimension sizes
  - `hw_spat_dim` — spatial dimension sizes
  - Explain in plain language what the mapping does: e.g. "flattens batch and width into the temporal axis while keeping channel-spatial as the per-cycle data word"

- **If AddressMapping is used**: show the full mapping with:
  - `base` address
  - `hw_temp_einops` — the temporal dimension ordering for address generation
  - `hw_temp_dim` — temporal dimension sizes
  - `hw_temp_stride` — stride for each temporal dimension
  - Explain the resulting address pattern: e.g. "row-major layout with batch as the outermost loop, stride 8 per batch"

A good dimension description for test data looks like:
> Input tensor shape `(2, 4, 8)` interpreted as `(batch=2, width=4, channel=8)` in `torch.int16`. The TileMapping splits channel into `ct=2` (temporal) and `cs=4` (spatial per cycle), giving a hardware view of `(batch * width * ct, cs) = (16, 4)` — 16 clock cycles, 4 elements per cycle.

---

## Report Template

The output report **must** follow this structure. Do not reorder or omit sections.

```markdown
# Testbench Report: <DUT Module Name>

## 1. DUT Interface Diagram

(ASCII art block diagram of DUT ports with signal names, bit widths, and grouping)

## 2. Testbench Topology

(ASCII art topology diagram showing all torchbit components, data flow direction,
 frontdoor/backdoor distinction, and golden model comparison point)

## 3. Stimulus Framework

For each interface group:
- Component name and class (e.g. `TwoPortBuffer(width=32, depth=256)`)
- Connected DUT signals
- Access method (backdoor / frontdoor, protocol description)

### Golden Model
- Algorithm (one sentence)
- Comparison method (e.g. `torch.equal()`, tolerance-based)

## 4. Test Data

### Data Layout Abstractions
- TileMapping: used / not used
- AddressMapping: used / not used
(If used, show full configuration with all parameters as described above)

### Data Regions
For each data region (source, destination, weights, etc.):
| Property   | Value                                    |
|------------|------------------------------------------|
| Semantic   | e.g. "source feature map"                |
| Shape      | e.g. `(batch=2, width=4, channel=8)`     |
| dtype      | e.g. `torch.int16`                       |
| Generation | e.g. `torch.arange()` / `torch.randn()` |
| Load method| e.g. `buffer.write()` / `backdoor_load_tensor()` |

## 5. Test Scenarios

| Test Name            | Purpose                          | Key Parameters         |
|----------------------|----------------------------------|------------------------|
| test_copy_basic      | Basic copy correctness           | 16 words, addr 0→128  |
| ...                  | ...                              | ...                    |
```

---

## What to Avoid

- Do NOT produce a line-by-line code walkthrough — summarize the architecture
- Do NOT skip the ASCII diagrams — they are the most valuable part
- Do NOT describe tensor shapes as bare tuples without dimension names
- Do NOT omit TileMapping/AddressMapping parameters when they are present in the code — these are the core configuration that defines the data layout
- Do NOT silently skip the TileMapping/AddressMapping usage statement — always explicitly report whether they are used or not
