"""
Verification components for hardware testing.

This module provides reusable verification components for building Cocotb
testbenches and verifying hardware designs.

Core Concepts:
1. Buffer/TwoPortBuffer: In-memory simulation buffers with HDL interface connectivity
   - Buffer: Basic memory buffer for read/write operations
   - TwoPortBuffer: Dual-port memory with async read/write, supports backpressure

2. BitStruct: Factory for bit-level struct manipulation
   - Creates structured views of integer values
   - Fields can be accessed as attributes for convenient R/W

3. TileMapping/AddressMapping: Tensor tiling and address mapping for memory arrays
   - TileMapping: Configures tensor-to-memory transformation with einops
   - AddressMapping: Maps multi-dimensional indices to flat memory addresses

4. Sender/PoolCollector/FIFOCollector: Data drivers and collectors for testbench
   - Sender: Drives data onto HDL signals with optional backpressure
   - PoolCollector: Collects data with valid signaling (no flow control)
   - FIFOCollector: Collects data with FIFO-style ready/empty flow control

5. InputPort/OutputPort: Signal wrapper abstractions for Cocotb
   - InputPort: Read wrapper handling None signals gracefully
   - OutputPort: Write wrapper with immediate value option

6. pad/depad: Tensor padding utilities for memory alignment
   NOTE: These have moved to torchbit.tiling module

Example:
    >>> from torchbit.tools import Buffer, Sender, TileMapping
    >>>
    >>> # Create a memory buffer
    >>> buf = Buffer(width=32, depth=1024)
    >>> buf.write(0x100, 0xDEADBEEF)
    >>> value = buf.read(0x100)
    >>>
    >>> # Create a data sender
    >>> sender = Sender(debug=True)
    >>> sender.connect(dut=dut, clk=dut.clk, data=dut.data, valid=dut.valid)
    >>> sender.load([0x1, 0x2, 0x3, 0x4])
    >>> cocotb.start_soon(sender.run())
    >>>
    >>> # Define tensor-to-memory mapping
    >>> mapping = TileMapping(
    ...     dtype=torch.float32,
    ...     sw_einops="h w -> h w",
    ...     hw_einops="h w -> (h w)",
    ...     hw_temp_dim={"h": 4},
    ...     hw_spat_dim={"w": 16},
    ...     base_addr=0x1000,
    ...     strides={"h": 64, "w": 4}
    ... )

Typical workflow:
    1. Define TileMapping for tensor-to-memory layout
    2. Initialize Buffer and connect to HDL via TwoPortBuffer
    3. Load input data using Sender
    4. Run simulation
    5. Collect results using PoolCollector/FIFOCollector
    6. Compare with golden model using torchbit.debug.compare()
"""
from .buffer import *
from .port import *
from .bit_struct import *
from .sender_collector import *

# Re-export tiling module for backward compatibility
from ..tiling import *
