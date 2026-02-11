"""
Verification components for hardware testing.

This module provides reusable verification components for building Cocotb
testbenches and verifying hardware designs.

Core Concepts:
1. Buffer/TwoPortBuffer: In-memory simulation buffers with HDL interface connectivity
   - Buffer: Basic memory buffer for read/write operations
   - TwoPortBuffer: Dual-port memory with async read/write, supports backpressure

2. BitStruct: Factory for bit-level struct manipulation (from torchbit.core)
   - Creates structured views of integer values
   - Fields can be accessed as attributes for convenient R/W

3. TileMapping/AddressMapping: Tensor tiling and address mapping for memory arrays
   - TileMapping: Converts software tensors to/from hardware vector sequences with einops
   - AddressMapping: Maps multi-dimensional indices to flat memory addresses
   - ContiguousAddressMapping: Row-major contiguous address mapping (strides auto-computed)

4. Driver/PoolMonitor/FIFOMonitor: Legacy data drivers and monitors for testbench
   - Driver: Drives data onto HDL signals with optional backpressure
   - PoolMonitor: Collects data with valid signaling (no flow control)
   - FIFOMonitor: Collects data with FIFO-style ready/empty flow control

5. FIFODriver/FIFOReceiver: Rich FIFO interface components with strategies
   - FIFODriver: Push data into DUT input FIFO (TB → DUT) with explicit polarity
   - FIFOReceiver: Capture data from DUT output FIFO (DUT → TB) with strategy

6. TransferStrategy: Pluggable transfer timing strategies
   - GreedyStrategy: Transfer whenever channel is ready (default)
   - RandomBackpressure: Randomly stall with configurable probability
   - BurstStrategy: Send N items, pause M cycles, repeat
   - ThrottledStrategy: Transfer at most once every N cycles

7. InputPort/OutputPort: Signal wrapper abstractions for Cocotb
   - InputPort: Read wrapper handling None signals gracefully
   - OutputPort: Write wrapper with immediate value option

8. pad/depad: Tensor padding utilities for memory alignment
   NOTE: These have moved to torchbit.tiling module

Example:
    >>> from torchbit.tools import Buffer, Driver, TileMapping
    >>>
    >>> # Create a memory buffer
    >>> buf = Buffer(width=32, depth=1024)
    >>> buf.write(0x100, 0xDEADBEEF)
    >>> value = buf.read(0x100)
    >>>
    >>> # Create a data driver
    >>> driver = Driver(debug=True)
    >>> driver.connect(dut=dut, clk=dut.clk, data=dut.data, valid=dut.valid)
    >>> driver.load([0x1, 0x2, 0x3, 0x4])
    >>> cocotb.start_soon(driver.run())
    >>>
    >>> # Define tensor-to-memory mapping
    >>> mapping = TileMapping(
    ...     dtype=torch.float32,
    ...     sw_einops="h w",
    ...     hw_einops="h w",
    ...     hw_temp_dim={"h": 4},
    ...     hw_spat_dim={"w": 16},
    ... )
    >>> addr_mapping = AddressMapping(
    ...     base=0x1000,
    ...     hw_temp_einops="h",
    ...     hw_temp_dim={"h": 4},
    ...     hw_temp_stride={"h": 16},
    ... )

Typical workflow:
    1. Define TileMapping for tensor-to-memory layout
    2. Initialize Buffer and connect to HDL via TwoPortBuffer
    3. Load input data using Driver
    4. Run simulation
    5. Collect results using PoolMonitor/FIFOMonitor
    6. Compare with golden model using torchbit.debug.compare()
"""
from .buffer import *
from .port import *
from ..core.bit_struct import *
from .driver import *
from .monitor import *
from .strategy import *
from .fifo import *

# Re-export tiling module for backward compatibility
from ..tiling import *
