"""
DUT Wrapper for MemoryMover with TwoPortBuffer

This wrapper encapsulates the MemoryMover DUT and uses TwoPortBuffer
directly as the memory (no separate Verilog RAM module needed).

A single TwoPortBuffer instance is used for both reading (source) and
writing (destination), demonstrating how TwoPortBuffer can act as a
shared memory for the DUT.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from torchbit.tools import TwoPortBuffer
from torchbit.core.vector import Vector
import torch


class MemoryMoverWrapper:
    """Wrapper for MemoryMover DUT with TwoPortBuffer.

    The wrapper uses a single TwoPortBuffer as shared memory:
    - Read port: For reading source data
    - Write port: For writing destination data

    This allows the test to initialize memory with tensor data and
    verify results after the DUT operates.
    """

    def __init__(self, dut):
        """Initialize the wrapper with DUT instance.

        Args:
            dut: The DUT instance (memory_mover)
        """
        self.dut = dut
        self.clock = Clock(dut.clk, 10, unit="ns")

        # Create a single TwoPortBuffer for both read and write
        # This acts as the shared memory that the DUT operates on
        self.buffer = TwoPortBuffer(width=32, depth=256, debug=False)

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

        MemoryMover DUT signals:
        - src_rd_addr, src_rd_csb, src_rd_dout, src_rd_dout_vld -> Read port
        - dst_wr_addr, dst_wr_csb, dst_wr_din -> Write port

        TwoPortBuffer.connect() signature:
            dut, clk, wr_csb, wr_din, wr_addr, rd_csb, rd_addr, rd_dout, rd_dout_vld
        """
        self.buffer.connect(
            dut=self.dut,
            clk=self.dut.clk,
            # Write port signals (for destination)
            wr_csb=self.dut.dst_wr_csb,
            wr_din=self.dut.dst_wr_din,
            wr_addr=self.dut.dst_wr_addr,
            # Read port signals (for source)
            rd_csb=self.dut.src_rd_csb,
            rd_addr=self.dut.src_rd_addr,
            rd_dout=self.dut.src_rd_dout,
            rd_dout_vld=self.dut.src_rd_dout_vld
        )

    def load_source_data(self, data: torch.Tensor, base_addr: int = 0):
        """Load source data into memory using TwoPortBuffer.

        This is a backdoor write operation - data is written directly
        to the buffer (which acts as memory) without going through the DUT.

        Args:
            data: PyTorch tensor containing source data
            base_addr: Base address for the data
        """
        for i, value in enumerate(data):
            addr = base_addr + i
            # Convert tensor element to integer and write
            int_value = int(value.item())
            self.buffer.write(addr, int_value)
            if self.buffer.debug:
                self.dut._log.info(f"[Wrapper] Backdoor write: {int_value} to addr {addr}")

    async def run_copy(self, src_base: int, dst_base: int, num_words: int):
        """Execute the copy operation on the DUT.

        Args:
            src_base: Source base address
            dst_base: Destination base address
            num_words: Number of words to copy
        """
        # Set up parameters
        self.dut.src_base_addr.value = src_base
        self.dut.dst_base_addr.value = dst_base
        self.dut.num_words.value = num_words
        self.dut.start.value = 0

        await ClockCycles(self.dut.clk, 2)

        # Start copy
        self.dut.start.value = 1
        await ClockCycles(self.dut.clk, 1)
        self.dut.start.value = 0

        # Wait for completion (with timeout protection)
        for _ in range(1000):
            await ClockCycles(self.dut.clk, 1)
            if self.dut.done.value == 1:
                break

        if self.dut.done.value != 1:
            raise TimeoutError("Memory mover timed out!")

    def read_destination_data(self, num_words: int, base_addr: int = 0) -> torch.Tensor:
        """Read destination data from memory using TwoPortBuffer.

        This is a backdoor read operation - data is read directly
        from the buffer (which acts as memory) without going through the DUT.

        Args:
            num_words: Number of words to read
            base_addr: Base address for reading

        Returns:
            PyTorch tensor containing the data
        """
        result = []
        for i in range(num_words):
            addr = base_addr + i
            int_value = self.buffer.read(addr)
            # Convert integer to tensor element
            result.append(int_value)
        return torch.tensor(result, dtype=torch.int32)
