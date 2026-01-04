"""
Memory buffer simulation for hardware verification.

Provides Buffer and TwoPortBuffer classes for simulating memory arrays
with HDL interface connectivity. Essential for modeling SRAM, register
files, and other memory-mapped storage during verification.
"""
import cocotb
from cocotb.triggers import RisingEdge, Timer
import numpy as np
from ..core.vector import Vector
from ..core.dtype import dtype_to_bits
from .port import InputPort, OutputPort
from .mapping import TileMapping
from ..utils.bit_ops import replicate_bits
import torch
import einops
import copy


class Buffer:
    """In-memory buffer for hardware verification.

    A simulation-level memory buffer that can be initialized from tensors
    and connected to HDL interfaces. Useful for modeling SRAM, register files,
    or any memory-mapped storage during verification.

    The buffer provides:
    - Random read/write access at integer addresses
    - Initialization from 2D matrices
    - Dump to tensors for comparison with golden models
    - Integration with TileMapping for complex tensor layouts

    Attributes:
        width (int): Data width in bits.
        depth (int): Number of entries (must be power of 2).
        addr_width (int): Address width in bits (log2 of depth).
        content (list): Internal storage for buffer values (integers).

    Raises:
        AssertionError: If depth is not a power of 2.

    Example:
        >>> from torchbit.tools import Buffer
        >>> import torch
        >>>
        >>> # Create a 32-bit wide, 1024-entry buffer
        >>> buf = Buffer(width=32, depth=1024)
        >>>
        >>> # Write and read values
        >>> buf.write(0x100, 0xDEADBEEF)
        >>> value = buf.read(0x100)
        >>> # value == 0xDEADBEEF
        >>>
        >>> # Initialize from matrix
        >>> mat = torch.randn(256, 4)  # 256 rows, 4 elements per row
        >>> buf.init_from_matrix(0, 256, mat)
        >>>
        >>> # Dump back to tensor
        >>> result = buf.dump_to_matrix(0, 256, torch.float32)

    Integration with TileMapping:
        >>> from torchbit.tools import Buffer, TileMapping
        >>> buf = Buffer(width=128, depth=1024)
        >>> mapping = TileMapping(
        ...     dtype=torch.float32,
        ...     sw_einops="c h w -> c h w",
        ...     hw_einops="c h w -> (c h w)",
        ...     hw_temp_dim={"c": 3},
        ...     hw_spat_dim={"h": 32, "w": 32},
        ...     base_addr=0,
        ...     strides={"c": 1024, "h": 32, "w": 4}
        ... )
        >>> tensor = torch.randn(3, 32, 32)
        >>> buf.init_from_tensor(tensor, mapping)
    """

    def __init__(self, width: int, depth: int):
        """Initialize a Buffer with specified width and depth.

        Args:
            width: Data width in bits.
            depth: Number of entries. Must be a power of 2 (e.g., 64, 128, 256...).

        Raises:
            AssertionError: If depth is not a power of 2.
        """
        # width unit: bit
        exp2 = 2 ** (3 + np.arange(20))
        assert depth in exp2.astype(int).tolist(), "depth must be power of 2"
        self.addr_width = np.log2(depth).astype(int)
        self.width = width
        self.depth = depth
        self.clear()

    def write(self, addr: int, int_value: int) -> None:
        """Write a value to the buffer at the specified address.

        Args:
            addr: Integer address (0 <= addr < depth).
            int_value: Integer value to store.
        """
        self.content[addr] = int_value

    def read(self, addr: int) -> int:
        """Read a value from the buffer at the specified address.

        Args:
            addr: Integer address (0 <= addr < depth).

        Returns:
            The integer value stored at the address.
        """
        return self.content[addr]

    def clear(self) -> None:
        """Clear all buffer contents to zero."""
        self.content = [0] * self.depth

    def init_from_matrix(self, addr_start: int, addr_end: int, matrix: torch.Tensor) -> None:
        """Initialize buffer from a 2D matrix.

        Each row of the matrix is converted to a packed integer and written
        to consecutive addresses.

        Args:
            addr_start: Starting address (inclusive).
            addr_end: Ending address (exclusive).
            matrix: 2D PyTorch tensor where each row becomes one entry.

        Raises:
            AssertionError: If matrix is not 2D.

        Note:
            The matrix row count should equal (addr_end - addr_start).
        """
        # [addr_start, addr_end)
        assert len(matrix.size()) == 2
        for i in range(addr_start, addr_end):
            self.content[i] = Vector.from_tensor(matrix[i - addr_start]).to_cocotb() & (
                (1 << self.width) - 1
            )

    def dump_to_matrix(self, addr_start: int, addr_end: int, dtype: torch.dtype) -> torch.Tensor:
        """Dump buffer contents to a 2D tensor.

        Reads consecutive addresses and converts each to a tensor row.

        Args:
            addr_start: Starting address (inclusive).
            addr_end: Ending address (exclusive). Use -1 for remaining entries.
            dtype: PyTorch dtype for the resulting tensor elements.

        Returns:
            A 2D PyTorch tensor with shape (num_entries, elements_per_entry).

        Note:
            elements_per_entry = width / bits_per_element(where bits_per_element
            is determined by dtype).
        """
        # [addr_start, addr_end)

        if addr_end == self.depth:
            addr_end = -1

        sel_content = copy.deepcopy(self.content[addr_start:addr_end])
        num_bit = dtype_to_bits(dtype)

        content_tensor = []
        for data in sel_content:
            content_tensor.append(
                Vector.from_int(
                    value_int=data, num=self.width // num_bit, dtype=dtype
                ).to_tensor()
            )
        return torch.stack(content_tensor, dim=0)

    def init_from_tensor(self, tensor: torch.Tensor, mapping: TileMapping) -> None:
        """Initialize buffer from a tensor using TileMapping.

        This is the primary method for loading tensor data into the buffer
        with complex layouts (tiling, striding, etc.).

        Args:
            tensor: Input PyTorch tensor of any shape.
            mapping: TileMapping defining the tensor-to-memory transformation.

        Note:
            Uses einops to rearrange tensor according to sw_einops/hw_einops
            formulas before writing to memory.
        """
        addr_list = mapping.address_mapping.get_addr_list()
        tensor_seq = einops.rearrange(
            tensor,
            mapping.sw_to_hw_formula,
            **mapping.hw_temp_dim,
            **mapping.hw_spat_dim,
        )  # [N, M]

        for tensor_row, addr in zip(tensor_seq, addr_list):
            self.write(addr, Vector.from_tensor(tensor_row).to_cocotb())

    def dump_to_tensor(self, mapping: TileMapping) -> torch.Tensor:
        """Dump buffer contents to a tensor using TileMapping.

        Reads memory according to the TileMapping and rearranges the data
        back to the original tensor layout.

        Args:
            mapping: TileMapping defining the memory-to-tensor transformation.

        Returns:
            A PyTorch tensor with the shape defined by the TileMapping.
        """
        addr_list = mapping.address_mapping.get_addr_list()
        cocotb_seq = [self.read(addr) for addr in addr_list]
        tensor_seq = [
            Vector.from_cocotb(int_value, mapping.num, mapping.dtype).to_tensor()
            for int_value in cocotb_seq
        ]
        tensor = einops.rearrange(
            tensor_seq,
            mapping.hw_to_sw_formula,
            **mapping.hw_temp_dim,
            **mapping.hw_spat_dim,
        )  # [N, M]
        return tensor


def is_trig(value: int, is_pos_trig: bool) -> bool:
    """Check if a signal value represents a valid trigger.

    Args:
        value: Integer signal value (0 or 1 for binary signals).
        is_pos_trig: If True, treat non-zero as trigger; if False, treat zero as trigger.

    Returns:
        True if the trigger condition is met, False otherwise.
    """
    is_pos = value != 0
    return is_pos if is_pos_trig else not is_pos


class TwoPortBuffer(Buffer):
    """Dual-port memory buffer with async read/write capability.

    Extends Buffer with Cocotb interface connectivity for driving
    and monitoring HDL signals. Supports:
    - Independent read and write ports
    - Optional backpressure (ready/valid)
    - Optional write masking
    - Configurable clock edge triggering

    The read port is always ready and outputs data immediately when
    read chip select is active. The write port supports optional
    backpressure and write masking.

    Attributes:
        bp (bool): Enable backpressure signaling.
        wrmask (bool): Enable write masking.
        debug (bool): Enable debug logging.
        is_pos_trig (bool): Use positive-edge triggering.

    Example:
        >>> from torchbit.tools import TwoPortBuffer
        >>> buf = TwoPortBuffer(width=32, depth=1024, backpressure=True)
        >>> buf.connect(
        ...     dut=dut, clk=dut.clk,
        ...     wr_csb=dut.wr_csb, wr_din=dut.wr_din,
        ...     wr_addr=dut.wr_addr, rd_csb=dut.rd_csb,
        ...     rd_addr=dut.rd_addr, rd_dout=dut.rd_dout,
        ...     rd_dout_vld=dut.rd_dout_vld,
        ...     wr_ready=dut.wr_ready, rd_ready=dut.rd_ready
        ... )
        >>> await buf.init()
        >>> cocotb.start_soon(buf.run())

    Signal Interface:
        Write Port:
            - wr_csb: Write chip select (active based on trigger)
            - wr_din: Write data input
            - wr_addr: Write address
            - wr_ready: (optional) Backpressure ready signal
            - wr_mask: (optional) Write mask for byte enables

        Read Port:
            - rd_csb: Read chip select (active based on trigger)
            - rd_addr: Read address
            - rd_dout: Read data output
            - rd_dout_vld: Read data valid
            - rd_ready: (optional) Backpressure ready signal
    """

    def __init__(
        self,
        width: int,
        depth: int,
        backpressure: bool = False,
        wrmask: bool = False,
        debug: bool = False,
        postive_trigger: bool = False,
    ):
        """Initialize a TwoPortBuffer.

        Args:
            width: Data width in bits.
            depth: Number of entries (must be power of 2).
            backpressure: Enable ready/valid flow control signals.
            wrmask: Enable write masking for byte enables.
            debug: Enable debug logging.
            postive_trigger: Use positive edge for chip select triggering.
        """
        # width as byte as unit
        super().__init__(width, depth)
        self.bp = backpressure
        self.debug = debug
        self.wrmask = wrmask
        self.is_pos_trig = postive_trigger

    def connect(
        self,
        dut,
        clk,
        wr_csb,
        wr_din,
        wr_addr,
        rd_csb,
        rd_addr,
        rd_dout,
        rd_dout_vld,
        wr_ready=None,
        rd_ready=None,
        wr_mask=None,
    ) -> None:
        """Connect the buffer to HDL signals.

        Args:
            dut: The DUT (Design Under Test) object.
            clk: Clock signal.
            wr_csb: Write chip select signal.
            wr_din: Write data input signal.
            wr_addr: Write address signal.
            rd_csb: Read chip select signal.
            rd_addr: Read address signal.
            rd_dout: Read data output signal.
            rd_dout_vld: Read data valid signal.
            wr_ready: (optional) Write ready/backpressure signal.
            rd_ready: (optional) Read ready/backpressure signal.
            wr_mask: (optional) Write mask signal for byte enables.
        """
        self.dut = dut
        self.clk = clk
        self.wr_csb = InputPort(wr_csb)
        self.wr_din = InputPort(wr_din)
        self.wr_addr = InputPort(wr_addr)
        self.rd_csb = InputPort(rd_csb)
        self.rd_addr = InputPort(rd_addr)
        self.rd_dout = OutputPort(rd_dout)
        self.rd_dout_vld = OutputPort(rd_dout_vld)
        if self.bp:
            self.wr_ready = OutputPort(wr_ready, set_immediately=True)
            self.rd_ready = OutputPort(rd_ready, set_immediately=True)
        if self.wrmask:
            self.wr_mask = InputPort(wr_mask)

    async def init(self) -> None:
        """Initialize output signals to idle state.

        Call this method before running the buffer to ensure all
        outputs are in a known idle state.
        """
        self.rd_dout_vld.set(0)
        if self.bp:
            self.wr_ready.set(0)
            self.rd_ready.set(0)

    async def _run_read(self) -> None:
        """Background task: handle read operations.

        Continuously monitors read chip select and outputs data when
        the read is active. Runs until the simulation ends.
        """
        while True:
            await RisingEdge(self.clk)
            read_trig = is_trig(self.rd_csb.get(), self.is_pos_trig)

            if read_trig:
                addr = self.rd_addr.get()
                self.rd_dout.set(self.content[addr])
                if self.debug:
                    self.dut._log.info(f"[TWBUF] read {self.content[addr]} from {addr}")
                self.rd_dout_vld.set(1)
            else:
                self.rd_dout_vld.set(0)

            if self.bp:
                self.rd_ready.set(1 if read_trig else 0)

    async def _run_write(self) -> None:
        """Background task: handle write operations.

        Continuously monitors write chip select and writes data when
        the write is active. Supports optional write masking.
        Runs until the simulation ends.
        """
        while True:
            await RisingEdge(self.clk)
            write_trig = is_trig(self.wr_csb.get(), self.is_pos_trig)

            if self.bp:
                self.wr_ready.set(1 if write_trig else 0)

            if write_trig:
                addr = self.wr_addr.get() & ((1 << self.addr_width) - 1)
                data = self.wr_din.get() & ((1 << self.width) - 1)
                if self.wrmask:
                    wr_mask = replicate_bits(8, self.wr_mask.get())
                    data = (data & wr_mask) | (self.content[addr] & ~wr_mask)
                if self.debug:
                    self.dut._log.info(
                        f"[TWBUF] write {data}/{self.wr_din.get()} to {addr}"
                    )

                await Timer(1, "step")
                self.content[addr] = data

    async def run(self) -> None:
        """Start the read and write background tasks.

        This method launches the read and write coroutines as concurrent
        tasks. Call this after init() to start the buffer operation.

        Example:
            >>> buf = TwoPortBuffer(width=32, depth=1024)
            >>> buf.connect(...)
            >>> await buf.init()
            >>> cocotb.start_soon(buf.run())
        """
        cocotb.start_soon(self._run_read())
        cocotb.start_soon(self._run_write())
