"""
Core tensor conversion utilities for translating between PyTorch tensors and
Cocotb/HDL formats.

This module provides the fundamental building blocks for hardware verification:

1. Vector: 1D tensor conversion (scalar and array inputs)
   - Converts PyTorch tensors to/from integer values for HDL interfaces
   - Supports Cocotb LogicArray and int value conversion
   - Handles data packing/unpacking for multi-element tensors

2. VectorSequence: 2D tensor conversion (memory-mapped interfaces)
   - Reads/writes memory hex files in $readmemh format
   - Reads/writes binary files for raw memory data
   - Converts 2D tensors to/from memory file representations

3. Dtype: Type mappings and bit-width definitions
   - Maps PyTorch/numpy dtypes to bit widths
   - Provides standard dtype conversions for bit-level operations

4. IntSequence: Typed integer sequence for hardware verification data

5. BitStruct/BitField: Bit-level struct manipulation
   - Creates structured views of integer values
   - Fields can be accessed as attributes for convenient R/W

All classes support conversion between:
- PyTorch tensors and Cocotb LogicArray values
- PyTorch tensors and memory hex/binary files
- Integer representations for HDL signal driving

Example:
    >>> import torch
    >>> from torchbit.core import Vector, VectorSequence
    >>>
    >>> # 1D tensor conversion
    >>> tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    >>> vec = Vector.from_tensor(tensor)
    >>> cocotb_value = vec.to_cocotb()
    >>>
    >>> # 2D tensor for memory files
    >>> vs = VectorSequence.from_tensor(torch.randn(256, 4))
    >>> vs.to_memhexfile("memory.hex")
    >>> vs_loaded = VectorSequence.from_memhexfile("memory.hex", torch.float32)

Typical usage in Cocotb tests:
    >>> @cocotb.test
    >>> async def test_dut(dut):
    >>>     # Drive input from tensor
    >>>     tensor = torch.tensor([1.5, 2.5], dtype=torch.float32)
    >>>     dut.data_in.value = tensor_to_cocotb(tensor)
    >>>
    >>>     # Read output to tensor
    >>>     await RisingEdge(dut.clk)
    >>>     result = cocotb_to_tensor(dut.data_out.value, 8, torch.float32)
"""
from .dtype import *
from .vector import *
from .vector_sequence import *
from .logic_sequence import *
from .bit_struct import *
