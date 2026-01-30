# Example 06: Tile Mapping with Spatial Mapping

This example demonstrates a tile transpose operation using a buffer move unit connected to TwoPortBuffer. The operation transforms data from `(b, w, c) * (cs)` layout to `(w, b, c) * (cs)` layout, where `cs` is the spatial dimension.

## Overview

### Spatial Mapping Concept

**Spatial Dimension (cs)**: Number of elements transferred per hardware cycle
- DATA_WIDTH = 64 bits (hardware interface width)
- dtype = int16 (16 bits per element)
- cs = 64 / 16 = 4 (4 elements transferred per cycle)

**Temporal Dimension (ct)**: Number of cycles needed for a dimension
- If c = 8, then ct = c / cs = 8 / 4 = 2 (temporal cycles)

### Data Layout Transformation

```
Source Layout (b=2, w=4, c=8, cs=4):
Software view: (b, w, c) = (2, 4, 8)
- c is split: c = ct * cs = 2 * 4
Memory layout: (b, w, ct) with cs spatial elements per address
- sw_einops: "b w (ct cs) -> b w (ct cs)"
- hw_einops: "b w (ct cs) -> (b w ct) cs"
- hw_temp_dim: {b: 2, w: 4, ct: 2}
- hw_spat_dim: {cs: 4}

Destination Layout (w=4, b=2, c=8, cs=4):
Software view: (w, b, c) = (4, 2, 8) - b and w swapped!
Memory layout: (w, b, ct) with cs spatial elements per address
- sw_einops: "w b (ct cs) -> w b (ct cs)"
- hw_einops: "w b (ct cs) -> (w b ct) cs"
- hw_temp_dim: {w: 4, b: 2, ct: 2}
- hw_spat_dim: {cs: 4}
```

### Key Features

- **Spatial Mapping**: cs=4 elements transferred per cycle (parallel transfer)
- **Temporal Mapping**: ct dimension requires multiple cycles
- **Instruction-based configuration**: b, w, c dimensions and base addresses configurable at runtime
- **Non-overlapping regions**: Source and destination regions are at non-overlapping addresses
- **TwoPortBuffer integration**: Uses `init_from_tensor` and `dump_to_tensor` for backdoor access

## Directory Structure

```
06_tile_mapping/
├── run.py                          # Entry point
├── src/rtl/
│   └── tile_mover.v               # DUT - (b,w,c) -> (w,b,c) transpose
├── tb/
│   ├── dut_tile_mover/
│   │   ├── wrapper/
│   │   │   └── tile_mover_wrapper.py  # DUT wrapper with spatial mapping
│   │   └── tests/
│   │       ├── test_transpose_basic.py       # Basic (2,4,8) transpose
│   │       ├── test_transpose_nonoverlap.py  # Non-overlap verification
│   │       └── test_transpose_configurable.py # Various dimensions
│   └── common/
│       └── golden_model/
│           └── tile_mover_golden.py  # PyTorch einops reference model
└── README.md
```

## Running the Example

```bash
# From the examples directory
cd 06_tile_mapping
python run.py
```

## TileMover DUT Interface

```systemverilog
module tile_mover #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 64
)(
    input  wire clk,
    input  wire rst_n,

    // Instruction port
    input  wire [ADDR_WIDTH-1:0] src_base_addr,
    input  wire [ADDR_WIDTH-1:0] dst_base_addr,
    input  wire [7:0] b_dim,         // B dimension (batch)
    input  wire [7:0] w_dim,         // W dimension (width)
    input  wire [7:0] c_dim,         // C dimension (channel = ct * cs)
    input  wire start,
    output reg  done,

    // TwoPortBuffer connection
    output wire [ADDR_WIDTH-1:0] rd_addr,
    output wire rd_csb,
    input  wire [DATA_WIDTH-1:0] rd_dout,
    input  wire rd_dout_vld,
    output wire [ADDR_WIDTH-1:0] wr_addr,
    output wire [DATA_WIDTH-1:0] wr_din,
    output wire wr_csb
);
```

## Usage Example

```python
# Create source tensor (b=2, w=4, c=8)
b, w, c = 2, 4, 8
src_tensor = torch.arange(b * w * c, dtype=torch.int16).reshape(b, w, c)

# Load into memory at base address 0x00 using init_from_tensor
wrapper.load_source_tensor(src_tensor, b, w, c, base_addr=0x00)

# Run transpose to base address 0x40
await wrapper.run_transpose(src_base=0x00, dst_base=0x40, b=b, w=w, c=c)

# Read transposed result (w, b, c) using dump_to_tensor
result = wrapper.read_destination_tensor(b, w, c, base_addr=0x40)
# result.shape == (4, 2, 8) - b and w swapped!
```

## TileMapping Configuration

**Source mapping** (b, w, c layout):
```python
ct = c // CS  # temporal dimension
src_mapping = TileMapping(
    dtype=torch.int16,
    sw_einops="b w (ct cs) -> b w (ct cs)",
    hw_einops="b w (ct cs) -> (b w ct) cs",
    hw_temp_dim={"b": b, "w": w, "ct": ct},
    hw_spat_dim={"cs": CS},
    base_addr=0x00,
    strides={"b": w * ct, "w": ct}
)
```

**Destination mapping** (w, b, c layout - transposed):
```python
dst_mapping = TileMapping(
    dtype=torch.int16,
    sw_einops="w b (ct cs) -> w b (ct cs)",
    hw_einops="w b (ct cs) -> (w b ct) cs",
    hw_temp_dim={"w": w, "b": b, "ct": ct},
    hw_spat_dim={"cs": CS},
    base_addr=0x40,
    strides={"w": b * ct, "b": ct}
)
```

## Tests

1. **test_transpose_basic.py**: Basic (2,4,8) -> (4,2,8) transpose
2. **test_transpose_nonoverlap.py**: Verifies non-overlapping source/destination regions
3. **test_transpose_configurable.py**: Tests various dimensions (b,w,c combinations)

## Key Classes

- `TileMoverWrapper`: Encapsulates DUT + TwoPortBuffer with spatial mapping
- `TileMoverGolden`: PyTorch reference using einops rearrange (pure software)
- `TileMapping`: Defines tensor-to-memory transformation with temporal/spatial dimensions
