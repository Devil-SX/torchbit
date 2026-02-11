# Example 06: Tile Mapping

Demonstrates TileMapping and AddressMapping for tensor-to-memory layout with spatial/temporal dimension mapping. The DUT performs a (b, w, c) -> (w, b, c) tile transpose using TwoPortBuffer as shared memory.

## Spatial Mapping Concept

**Spatial Dimension (cs)**: Number of elements transferred per hardware cycle
- DATA_WIDTH = 64 bits, dtype = int16 (16 bits per element)
- cs = 64 / 16 = 4 elements per cycle

**Temporal Dimension (ct)**: Number of cycles needed
- If c = 8, then ct = c / cs = 8 / 4 = 2 cycles

## Directory Structure

```
06_tile_mapping/
├── run.py                  # Entry point (requires cocotb + Verilator)
├── test_tile_mapping.py    # Cocotb tests
├── dut/
│   └── tile_mover.v        # DUT - (b,w,c) -> (w,b,c) transpose
└── README.md
```

## Running the Example

```bash
cd examples/06_tile_mapping
python run.py
```

## Key Classes

- **TileMapping**: Defines tensor-to-memory layout using einops notation
- **AddressMapping**: Maps multi-dimensional indices to flat memory addresses
- **TwoPortBuffer**: Shared memory with `backdoor_load_tensor` / `backdoor_dump_tensor`

## Usage

```python
from torchbit.tiling import TileMapping, AddressMapping

# Source mapping: (b, w, c) layout with spatial dimension cs=4
ct = c // CS
src_mapping = TileMapping(
    dtype=torch.int16,
    sw_einops="b w (ct cs)",
    hw_einops="(b w ct) cs",
    hw_temp_dim={"b": b, "w": w, "ct": ct},
    hw_spat_dim={"cs": CS},
)
src_addr = AddressMapping(
    base=0x00,
    hw_temp_einops="b w ct",
    hw_temp_dim={"b": b, "w": w, "ct": ct},
    hw_temp_stride={"b": w * ct, "w": ct, "ct": 1},
)

# Load tensor into buffer and run DUT
buffer.backdoor_load_tensor(tensor, src_mapping, src_addr)
```

## Tests

1. **test_transpose_basic**: Basic (2,4,8) -> (4,2,8) transpose with einops golden model
2. **test_transpose_configurable**: Multiple dimension configurations (1x2x4, 2x2x4, 2x4x8, etc.)
