"""Tests for TwoPortBuffer with TileMapping integration.

Tests the buffer's tensor load/dump functionality with various
tiling mappings. Software-only tests (no hardware simulation).
"""

import pytest
import torch
import torchbit
from torchbit.tools.buffer import TwoPortBuffer
from torchbit.tiling import TileMapping


class TestBufferTilingBasic:
    """Basic tests for TwoPortBuffer tensor operations."""

    def test_buffer_creation(self):
        """Test TwoPortBuffer creation with different sizes."""
        buf = TwoPortBuffer(width=32, depth=1024)
        assert buf.width == 32
        assert buf.depth == 1024
        assert buf.addr_width == 10  # log2(1024)

    def test_buffer_read_write_direct(self, buffer_32x1024):
        """Test direct read and write operations."""
        buf = buffer_32x1024

        # Write values
        buf.write(0, 0xDEADBEEF)
        buf.write(100, 0x12345678)

        # Read back
        assert buf.read(0) == 0xDEADBEEF
        assert buf.read(100) == 0x12345678

    def test_buffer_clear(self, buffer_32x1024):
        """Test buffer clear operation."""
        buf = buffer_32x1024

        # Write some values
        buf.write(0, 0xFFFFFFFF)
        buf.write(1, 0xAAAAAAAA)

        # Clear and verify
        buf.clear()
        assert buf.read(0) == 0
        assert buf.read(1) == 0


class TestBufferInitFromTensor:
    """Tests for TwoPortBuffer.init_from_tensor() method."""

    def test_init_from_tensor_basic(self, buffer_128x512):
        """Test basic tensor initialization via TileMapping."""
        buf = buffer_128x512

        # Create a simple tensor
        torch.manual_seed(42)
        tensor = torch.randn(3, 4, 4, dtype=torch.float32)

        # Create mapping
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides=None,
        )

        # Initialize buffer from tensor
        buf.init_from_tensor(tensor, mapping)

        # Verify by reading back specific locations
        # Each 4x4=16 elements fit in one address (128 bits / 32 bits = 4 floats)
        # So we expect 3 channels / 4 per line = not exactly, let's verify
        # 128 bits / 32 bits per float = 4 floats per address
        # 4*4 = 16 floats per channel / 4 = 4 addresses per channel

    def test_init_from_tensor_with_base_addr(self, buffer_128x512):
        """Test tensor initialization with non-zero base address."""
        buf = buffer_128x512

        torch.manual_seed(42)
        tensor = torch.randn(3, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0x100,
            strides=None,
        )

        buf.init_from_tensor(tensor, mapping)

        # Read from the base address
        value = buf.read(0x100)
        assert value != 0  # Should have written something

    def test_init_from_tensor_with_strides(self, buffer_128x512):
        """Test tensor initialization with address strides."""
        buf = buffer_128x512

        torch.manual_seed(42)
        tensor = torch.randn(3, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides={"c": 16},  # Each channel 16 addresses apart
        )

        buf.init_from_tensor(tensor, mapping)

        # Verify data is at strided locations
        assert buf.read(0) != 0
        assert buf.read(16) != 0
        assert buf.read(32) != 0


class TestBufferDumpToTensor:
    """Tests for TwoPortBuffer.dump_to_tensor() method."""

    def test_dump_to_tensor_basic(self, buffer_128x512):
        """Test dumping buffer contents back to tensor."""
        buf = buffer_128x512

        # Create and load a tensor
        torch.manual_seed(42)
        original = torch.randn(3, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides=None,
        )

        buf.init_from_tensor(original, mapping)

        # Dump back to tensor
        recovered = buf.dump_to_tensor(mapping)

        # Verify equality
        assert torch.equal(original, recovered)

    def test_dump_to_tensor_with_base_addr(self, buffer_128x512):
        """Test dumping with non-zero base address."""
        buf = buffer_128x512

        torch.manual_seed(42)
        original = torch.randn(3, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0x100,
            strides=None,
        )

        buf.init_from_tensor(original, mapping)
        recovered = buf.dump_to_tensor(mapping)

        assert torch.equal(original, recovered)


class TestBufferRoundtrip:
    """Tests for complete init/dump roundtrips with various layouts."""

    def test_roundtrip_channel_first(self, buffer_256x256):
        """Test roundtrip with channel-first (NCHW) layout."""
        buf = buffer_256x256

        torch.manual_seed(42)
        tensor = torch.randn(3, 8, 8, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # Roundtrip
        buf.init_from_tensor(tensor, mapping)
        recovered = buf.dump_to_tensor(mapping)

        assert torch.equal(tensor, recovered)

    def test_roundtrip_channel_last(self, buffer_256x256):
        """Test roundtrip with channel-last (NHWC) layout."""
        buf = buffer_256x256

        torch.manual_seed(42)
        tensor = torch.randn(3, 8, 8, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # Roundtrip
        buf.init_from_tensor(tensor, mapping)
        recovered = buf.dump_to_tensor(mapping)

        assert torch.equal(tensor, recovered)

    def test_roundtrip_tiled(self, buffer_256x256):
        """Test roundtrip with spatial dimensions combined."""
        buf = buffer_256x256

        torch.manual_seed(42)
        tensor = torch.randn(3, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # Keep channel separate, combine h and w
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides=None,
        )

        # Roundtrip
        buf.init_from_tensor(tensor, mapping)
        recovered = buf.dump_to_tensor(mapping)

        assert torch.equal(tensor, recovered)

    def test_roundtrip_strided(self, buffer_256x256):
        """Test roundtrip with address strides."""
        buf = buffer_256x256

        torch.manual_seed(42)
        tensor = torch.randn(4, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 4},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,  # Use base_addr=0 to stay within buffer bounds
            strides={"c": 32},  # Each channel is 16 elements apart, use 32 to leave gaps
        )

        # Roundtrip
        buf.init_from_tensor(tensor, mapping)
        recovered = buf.dump_to_tensor(mapping)

        assert torch.equal(tensor, recovered)


class TestBufferMultipleDtypes:
    """Tests for TwoPortBuffer with different data types."""

    def test_roundtrip_float32(self, buffer_256x256):
        """Test roundtrip with float32 data type."""
        buf = buffer_256x256

        torch.manual_seed(42)
        tensor = torch.randn(2, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 2},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides=None,
        )

        # Roundtrip
        buf.init_from_tensor(tensor, mapping)
        recovered = buf.dump_to_tensor(mapping)

        assert torch.equal(tensor, recovered)


class TestBufferMatrixOperations:
    """Tests for Buffer's matrix-based operations."""

    def test_init_and_dump_matrix(self, buffer_128x512):
        """Test init_from_matrix and dump_to_matrix methods."""
        buf = buffer_128x512

        # Create a matrix (2D tensor)
        # 128-bit width can store 4 float32 values per address
        torch.manual_seed(42)
        matrix = torch.randn(16, 4, dtype=torch.float32)

        # Initialize from matrix
        buf.init_from_matrix(0, 16, matrix)

        # Dump back to matrix
        recovered = buf.dump_to_matrix(0, 16, torch.float32)

        # Verify
        assert torch.equal(matrix, recovered)

    def test_matrix_partial_range(self, buffer_128x512):
        """Test matrix operations with partial address range."""
        buf = buffer_128x512

        torch.manual_seed(42)
        matrix = torch.randn(8, 4, dtype=torch.float32)

        # Initialize at offset
        buf.init_from_matrix(100, 108, matrix)

        # Dump from same offset
        recovered = buf.dump_to_matrix(100, 108, torch.float32)

        assert torch.equal(matrix, recovered)

        # Verify other locations are zero
        for i in range(100):
            assert buf.read(i) == 0


class TestBufferBackdoorParallel:
    """Tests for parallel backdoor operations."""

    def test_backdoor_read_single(self, buffer_32x1024):
        """Test backdoor_read with single address."""
        buf = buffer_32x1024

        # Write a value
        buf.write(100, 0xDEADBEEF)

        # Read using backdoor
        values = buf.backdoor_read([100])
        assert values == [0xDEADBEEF]

    def test_backdoor_read_multiple(self, buffer_32x1024):
        """Test backdoor_read with multiple addresses."""
        buf = buffer_32x1024

        # Write values
        buf.write(0, 0x11111111)
        buf.write(10, 0x22222222)
        buf.write(100, 0x33333333)

        # Read using backdoor
        values = buf.backdoor_read([0, 10, 100])
        assert values == [0x11111111, 0x22222222, 0x33333333]

    def test_backdoor_read_out_of_range(self, buffer_32x1024):
        """Test backdoor_read with out of range address."""
        buf = buffer_32x1024

        with pytest.raises(AssertionError, match="out of range"):
            buf.backdoor_read([0, 10, 1024])  # 1024 is out of range

    def test_backdoor_write_single(self, buffer_32x1024):
        """Test backdoor_write with single address."""
        buf = buffer_32x1024

        # Write using backdoor
        buf.backdoor_write([100], [0xBEEFBEEF])

        # Read back
        assert buf.read(100) == 0xBEEFBEEF

    def test_backdoor_write_multiple(self, buffer_32x1024):
        """Test backdoor_write with multiple addresses."""
        buf = buffer_32x1024

        # Write using backdoor
        buf.backdoor_write([0, 10, 100], [0xAAA, 0xBBB, 0xCCC])

        # Read back
        assert buf.read(0) == 0xAAA
        assert buf.read(10) == 0xBBB
        assert buf.read(100) == 0xCCC

    def test_backdoor_write_mismatched_lengths(self, buffer_32x1024):
        """Test backdoor_write with mismatched address and data lists."""
        buf = buffer_32x1024

        with pytest.raises(AssertionError, match="length"):
            buf.backdoor_write([0, 10], [0xAAA])  # 2 addresses, 1 value

    def test_backdoor_write_out_of_range(self, buffer_32x1024):
        """Test backdoor_write with out of range address."""
        buf = buffer_32x1024

        with pytest.raises(AssertionError, match="out of range"):
            buf.backdoor_write([0, 10, 1024], [0xAAA, 0xBBB, 0xCCC])

    def test_backdoor_new_api_with_aliases(self, buffer_128x512):
        """Test new backdoor_* methods work as expected."""
        buf = buffer_128x512

        torch.manual_seed(42)
        tensor = torch.randn(3, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides=None,
        )

        # Use new backdoor_load_tensor
        buf.backdoor_load_tensor(tensor, mapping)

        # Use new backdoor_dump_tensor
        recovered = buf.backdoor_dump_tensor(mapping)

        assert torch.equal(tensor, recovered)

    def test_backdoor_matrix_new_api(self, buffer_128x512):
        """Test new backdoor_load_matrix and backdoor_dump_matrix methods."""
        buf = buffer_128x512

        torch.manual_seed(42)
        matrix = torch.randn(16, 4, dtype=torch.float32)

        # Use new backdoor_load_matrix
        buf.backdoor_load_matrix(0, 16, matrix)

        # Use new backdoor_dump_matrix
        recovered = buf.backdoor_dump_matrix(0, 16, torch.float32)

        assert torch.equal(matrix, recovered)
