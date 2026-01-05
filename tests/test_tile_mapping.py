"""Tests for TileMapping and AddressMapping classes.

Tests the tensor-to-memory mapping functionality without hardware simulation.
"""

import pytest
import torch
import numpy as np
from torchbit.tools.mapping import TileMapping, AddressMapping, tensor_to_cocotb_seq, cocotb_seq_to_tensor


class TestAddressMapping:
    """Tests for AddressMapping class."""

    def test_address_mapping_basic(self):
        """Test basic address mapping with row-major layout."""
        mapping = AddressMapping(
            base=0,
            strides=(16, 1),  # row-major: stride of 16 for rows, 1 for cols
            max_values=(4, 16),  # 4 rows, 16 columns
        )

        addrs = mapping.get_addr_list()
        assert len(addrs) == 4 * 16  # Total elements

        # Check first row addresses: 0, 1, 2, ...
        assert addrs[0] == 0
        assert addrs[1] == 1
        assert addrs[15] == 15

        # Check second row starts at 16
        assert addrs[16] == 16

    def test_address_mapping_base_offset(self):
        """Test address mapping with base offset."""
        mapping = AddressMapping(
            base=0x1000,
            strides=(4, 1),
            max_values=(2, 4),
        )

        addrs = mapping.get_addr_list()
        assert addrs[0] == 0x1000
        assert addrs[4] == 0x1004  # Next row

    def test_address_mapping_column_major(self):
        """Test column-major address mapping."""
        mapping = AddressMapping(
            base=0,
            strides=(1, 4),  # column-major: stride of 1 for rows, 4 for cols
            max_values=(4, 4),
        )

        addrs = mapping.get_addr_list()
        # np.ndindex iterates in row-major order: (0,0), (0,1), (0,2), ..., (1,0), ...
        # But the strides make addresses column-major
        # (0,0) -> 0*1 + 0*4 = 0
        # (0,1) -> 0*1 + 1*4 = 4
        # (1,0) -> 1*1 + 0*4 = 1
        assert addrs[0] == 0
        assert addrs[1] == 4  # (0,1) -> 4
        assert addrs[4] == 1  # (1,0) -> 1


class TestTileMapping:
    """Tests for TileMapping class."""

    def test_tile_mapping_simple_flatten(self, sample_tensor_chw):
        """Test simple CHW to linear memory flattening."""
        tensor = sample_tensor_chw  # Shape: (3, 8, 8)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (c h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # Convert tensor using the mapping
        seq = tensor_to_cocotb_seq(tensor, mapping)
        recovered = cocotb_seq_to_tensor(seq, mapping)

        # Verify roundtrip
        assert torch.equal(tensor, recovered)

    def test_tile_mapping_channel_last(self, sample_tensor_chw):
        """Test NCHW to NHWC (channel-last) layout conversion."""
        tensor = sample_tensor_chw  # Shape: (3, 8, 8)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (h w c)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # Convert and recover
        seq = tensor_to_cocotb_seq(tensor, mapping)
        recovered = cocotb_seq_to_tensor(seq, mapping)

        # Verify roundtrip
        assert torch.equal(tensor, recovered)

    def test_tile_mapping_tiled_layout(self):
        """Test 2D tiling pattern."""
        # Create a tensor: 3 channels, 4x4 spatial
        torch.manual_seed(42)
        tensor = torch.randn(3, 4, 4, dtype=torch.float32)

        # For tiling, we use a simpler pattern that matches the API
        # The temporal dim (c) stays separate, spatial dims (h, w) are combined
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> c (h w)",  # Keep channel separate, combine h and w
            hw_temp_dim={"c": 3},  # Iterate over channels
            hw_spat_dim={"h": 4, "w": 4},  # Spatial dimensions are contiguous
            base_addr=0,
            strides=None,
        )

        # Convert and recover
        seq = tensor_to_cocotb_seq(tensor, mapping)
        recovered = cocotb_seq_to_tensor(seq, mapping)

        # Verify roundtrip
        assert torch.equal(tensor, recovered)

    def test_tile_mapping_with_strides(self, sample_tensor_chw):
        """Test TileMapping with address strides."""
        tensor = sample_tensor_chw  # Shape: (3, 8, 8)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (c h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0x1000,
            strides={"c": 64},
        )

        # Check that address mapping is created
        assert mapping.address_mapping is not None
        assert mapping.address_mapping.base == 0x1000

        # Get address list and verify stride
        addrs = mapping.address_mapping.get_addr_list()
        assert len(addrs) == 3  # One address per channel

        # Each channel should be 64 addresses apart
        assert addrs[1] - addrs[0] == 64
        assert addrs[2] - addrs[1] == 64

    def test_tile_mapping_different_dtypes(self):
        """Test TileMapping with float32 data type."""
        tensor = torch.randn(2, 4, 4, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (c h w)",
            hw_temp_dim={"c": 2},
            hw_spat_dim={"h": 4, "w": 4},
            base_addr=0,
            strides=None,
        )

        seq = tensor_to_cocotb_seq(tensor, mapping)
        recovered = cocotb_seq_to_tensor(seq, mapping)

        assert torch.equal(tensor, recovered)

    def test_tensor_cocotb_seq_roundtrip(self, sample_tensor_chw):
        """Test tensor_to_cocotb_seq and cocotb_seq_to_tensor roundtrip."""
        tensor = sample_tensor_chw

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (h w c)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # Forward conversion
        seq = tensor_to_cocotb_seq(tensor, mapping)
        assert len(seq) == 3  # One per channel
        assert all(isinstance(x, int) for x in seq)

        # Reverse conversion
        recovered = cocotb_seq_to_tensor(seq, mapping)

        # Verify
        assert torch.equal(tensor, recovered)

    def test_tile_mapping_num_attribute(self):
        """Test that num attribute is calculated correctly."""
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (h w c)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # num should be product of hw_spat_dim values
        expected_num = 8 * 8
        assert mapping.num == expected_num

    def test_tile_mapping_formulas(self):
        """Test that sw_to_hw and hw_to_sw formulas are generated correctly."""
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w -> c h w",
            hw_einops="c h w -> (h w c)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
            base_addr=0,
            strides=None,
        )

        # sw_to_hw: input is 'c h w', output is temporal dims + combined spatial dims
        # Temporal: c, Spatial: h w combined -> 'c (h w)'
        assert mapping.sw_to_hw_formula == "c h w -> c (h w)"
        # hw_to_sw: reverse from 'c (h w)' back to 'c h w'
        assert mapping.hw_to_sw_formula == "c (h w) -> c h w"
