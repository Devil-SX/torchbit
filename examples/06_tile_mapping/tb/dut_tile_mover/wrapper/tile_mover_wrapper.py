"""
DUT Wrapper for TileMover with TwoPortBuffer

This wrapper encapsulates the TileMover DUT and uses TwoPortBuffer
directly as the memory (no separate Verilog RAM module needed).

Demonstrates Spatial Mapping:
- DATA_WIDTH = 64 bits (hardware interface width)
- dtype = int16 (16 bits per element)
- cs = DATA_WIDTH / dtype_bits = 64 / 16 = 4 (spatial dimension)

The c dimension is split into (ct, cs):
- ct: temporal dimension (requires multiple cycles)
- cs: spatial dimension (transferred in one cycle)

Uses init_from_tensor and dump_to_tensor for backdoor memory access.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from torchbit.tools import TwoPortBuffer
from torchbit.tiling import TileMapping, AddressMapping
from torchbit.core.dtype import dtype_to_bits
from torchbit.core.vector import Vector
import torch
import einops


class TileMoverWrapper:
    """Wrapper for TileMover DUT with TwoPortBuffer.

    The wrapper uses a single TwoPortBuffer as shared memory:
    - Read port: For reading source data
    - Write port: For writing destination data

    Demonstrates spatial mapping where cs elements are transferred
    per cycle (spatial dimension).
    """

    # Spatial dimension: number of elements per hardware transfer
    # DATA_WIDTH = 64 bits, dtype = int16 (16 bits) -> cs = 4
    CS = 4
    DTYPE = torch.int16  # 16-bit elements

    def __init__(self, dut, debug: bool = False):
        """Initialize the wrapper with DUT instance.

        Args:
            dut: The DUT instance (tile_mover)
            debug: Enable debug logging
        """
        self.dut = dut
        self.debug = debug
        self.clock = Clock(dut.clk, 10, unit="ns")

        # Create a single TwoPortBuffer for both read and write
        # DATA_WIDTH = 64 bits (cs=4 elements of 16-bit each)
        self.buffer = TwoPortBuffer(width=64, depth=256, debug=debug)

    async def init(self):
        """Initialize the DUT: reset sequence and clock startup."""
        self.connect()

        # Start clock
        cocotb.start_soon(self.clock.start())

        # Initialize buffer
        await self.buffer.init()

        # Start buffer task
        cocotb.start_soon(self.buffer.run())

        # Reset
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, 5)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 2)

    def connect(self):
        """Connect TwoPortBuffer to the DUT interfaces.

        TileMover DUT signals:
        - rd_addr, rd_csb, rd_dout, rd_dout_vld -> Read port (source)
        - wr_addr, wr_csb, wr_din -> Write port (destination)

        TwoPortBuffer.connect() signature:
            dut, clk, wr_csb, wr_din, wr_addr, rd_csb, rd_addr, rd_dout, rd_dout_vld
        """
        self.buffer.connect(
            dut=self.dut,
            clk=self.dut.clk,
            # Write port signals (for destination)
            wr_csb=self.dut.wr_csb,
            wr_din=self.dut.wr_din,
            wr_addr=self.dut.wr_addr,
            # Read port signals (for source)
            rd_csb=self.dut.rd_csb,
            rd_addr=self.dut.rd_addr,
            rd_dout=self.dut.rd_dout,
            rd_dout_vld=self.dut.rd_dout_vld
        )

    def load_source_tensor(self, tensor: torch.Tensor, b: int, w: int, c: int, base_addr: int = 0):
        """Load source tensor into memory using TwoPortBuffer.init_from_tensor.

        The tensor is loaded in (b, w, c) layout with spatial mapping.

        Software view: (b, w, c) where c = ct * cs
        Hardware view: (b, w, ct) with cs spatial elements per transfer

        Args:
            tensor: 3D PyTorch tensor with shape (b, w, c)
            b: Batch dimension
            w: Width dimension
            c: Channel dimension (must be multiple of cs)
            base_addr: Base address for the data
        """
        assert tensor.shape == (b, w, c), f"Tensor shape {tensor.shape} doesn't match (b={b}, w={w}, c={c})"
        assert c % self.CS == 0, f"c={c} must be multiple of cs={self.CS}"

        ct = c // self.CS  # temporal dimension

        # Source mapping: (b, w, c) -> memory layout
        # Software view: tensor is (b, w, c) where c = ct * cs
        # Memory layout: (b*w*ct, cs) - flat temporal index, spatial elements per address
        #
        # sw_einops: "b w (ct cs)" - software dimension layout
        # hw_einops: "(b w ct) cs" - hardware vector sequence: (temporal) (spatial)
        src_mapping = TileMapping(
            dtype=self.DTYPE,
            sw_einops="b w (ct cs)",
            hw_einops="(b w ct) cs",
            hw_temp_dim={"b": b, "w": w, "ct": ct},
            hw_spat_dim={"cs": self.CS},
        )

        src_addr_mapping = AddressMapping(
            base=base_addr,
            hw_temp_einops="b w ct",
            hw_temp_dim={"b": b, "w": w, "ct": ct},
            hw_temp_stride={"b": w * ct, "w": ct, "ct": 1},
        )

        # Convert to int16 and load using mapping
        tensor_int = tensor.to(self.DTYPE)
        self.buffer.init_from_tensor(tensor_int, src_mapping, src_addr_mapping)

        if self.debug:
            self.dut._log.info(f"[Wrapper] Loaded {b}x{w}x{c} tensor at base_addr={base_addr:#x}")
            self.dut._log.info(f"[Wrapper] Spatial mapping: ct={ct}, cs={self.CS}")
            self.dut._log.info(f"[Wrapper] Source tensor:\n{tensor}")

    def read_destination_tensor(self, b: int, w: int, c: int, base_addr: int = 0) -> torch.Tensor:
        """Read destination tensor from memory using TwoPortBuffer.

        The data is read in (w, b, c) layout with spatial mapping.

        Args:
            b: Batch dimension (swapped with w in destination)
            w: Width dimension (swapped with b in destination)
            c: Channel dimension (must be multiple of cs)
            base_addr: Base address for reading

        Returns:
            3D PyTorch tensor with shape (w, b, c)
        """
        assert c % self.CS == 0, f"c={c} must be multiple of cs={self.CS}"

        ct = c // self.CS  # temporal dimension

        # Read data from buffer using simple address range
        # Memory stores (w*b*ct) addresses, each with cs elements
        num_addrs = w * b * ct
        tensor_2d = torch.zeros(num_addrs, self.CS, dtype=self.DTYPE)

        for i in range(num_addrs):
            addr = base_addr + i
            int_value = self.buffer.read(addr)
            # Unpack the int value into cs elements
            vec = Vector.from_cocotb(int_value, self.CS, self.DTYPE)
            tensor_2d[i] = vec.to_tensor()

        # Reshape 2D (w*b*ct, cs) to 3D (w, b, c) using einops
        # First expand: (w*b*ct, cs) -> (w, b, ct, cs)
        # Then combine: (w, b, ct, cs) -> (w, b, c) where c = ct*cs
        result = einops.rearrange(tensor_2d, "(w b ct) cs -> w b (ct cs)", w=w, b=b, ct=ct, cs=self.CS)

        if self.debug:
            self.dut._log.info(f"[Wrapper] Read {w}x{b}x{c} tensor from base_addr={base_addr:#x}")
            self.dut._log.info(f"[Wrapper] Destination tensor:\n{result}")

        return result

    async def run_transpose(self, src_base: int, dst_base: int, b: int, w: int, c: int):
        """Execute the transpose operation on the DUT.

        Args:
            src_base: Source base address
            dst_base: Destination base address
            b: Batch dimension
            w: Width dimension
            c: Channel dimension
        """
        # Set up parameters
        self.dut.src_base_addr.value = src_base
        self.dut.dst_base_addr.value = dst_base
        self.dut.b_dim.value = b
        self.dut.w_dim.value = w
        self.dut.c_dim.value = c
        self.dut.start.value = 0

        await ClockCycles(self.dut.clk, 2)

        # Start transpose
        self.dut.start.value = 1
        await ClockCycles(self.dut.clk, 1)
        self.dut.start.value = 0

        # Wait for completion (with timeout protection)
        # Total elements: b * w * c
        # Each cycle transfers cs elements spatially
        total_elements = b * w * c
        timeout = (total_elements // self.CS) * 2 + 100
        for _ in range(timeout):
            await ClockCycles(self.dut.clk, 1)
            if self.dut.done.value == 1:
                break

        if self.dut.done.value != 1:
            raise TimeoutError(f"Tile mover timed out after {timeout} cycles!")

        if self.debug:
            self.dut._log.info(f"[Wrapper] Transpose completed: {b}x{w}x{c} from {src_base:#x} to {dst_base:#x}")
