"""
Golden Model for MemoryMover DUT

The golden model provides the reference implementation for memory copying.
It's a simple Python/PyTorch implementation that computes the expected result.
"""
import torch


class MemoryMoverGolden:
    """Golden model for MemoryMover DUT.

    This is a simple functional model - it doesn't model timing,
    queues, or pipelines. It just computes the correct result.
    """

    def __init__(self):
        """Initialize the golden model."""
        self.memory = torch.zeros(256, dtype=torch.int32)

    def load_data(self, data: torch.Tensor, base_addr: int = 0):
        """Load data into the golden model's memory.

        Args:
            data: PyTorch tensor containing data to load
            base_addr: Base address for loading
        """
        num_words = min(len(data), len(self.memory) - base_addr)
        self.memory[base_addr:base_addr + num_words] = data[:num_words]

    def copy(self, src_base: int, dst_base: int, num_words: int) -> torch.Tensor:
        """Perform the copy operation and return expected result.

        Args:
            src_base: Source base address
            dst_base: Destination base address
            num_words: Number of words to copy

        Returns:
            PyTorch tensor containing the expected destination data
        """
        # Copy source to destination (immediate operation)
        for i in range(num_words):
            self.memory[dst_base + i] = self.memory[src_base + i]

        # Return the destination region
        return self.memory[dst_base:dst_base + num_words].clone()
