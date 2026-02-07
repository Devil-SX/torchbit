"""Tests for TileMapping and AddressMapping classes.

Tests the tensor-to-memory mapping functionality without hardware simulation.
"""

import pytest
import torch
import numpy as np
from torchbit.tiling import TileMapping, AddressMapping, ContiguousAddressMapping, tensor_to_cocotb_seq, cocotb_seq_to_tensor
from torchbit.core.int_sequence import IntSequence


class TestAddressMapping:
    """Tests for AddressMapping class."""

    def test_address_mapping_basic(self):
        """Test basic address mapping with row-major layout."""
        mapping = AddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 4, "col": 16},
            hw_temp_stride={"row": 16, "col": 1},
        )

        addrs = mapping.get_addr_list()
        assert len(addrs) == 4 * 16  # Total elements
        assert isinstance(addrs, IntSequence)

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
            hw_temp_einops="row col",
            hw_temp_dim={"row": 2, "col": 4},
            hw_temp_stride={"row": 4, "col": 1},
        )

        addrs = mapping.get_addr_list()
        assert addrs[0] == 0x1000
        assert addrs[4] == 0x1004  # Next row

    def test_address_mapping_column_major(self):
        """Test column-major address mapping."""
        mapping = AddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 4, "col": 4},
            hw_temp_stride={"row": 1, "col": 4},
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


class TestContiguousAddressMapping:
    """Tests for ContiguousAddressMapping class."""

    def test_contiguous_2d_row_major(self):
        """Test 2D contiguous mapping equals explicit row-major."""
        contiguous = ContiguousAddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 4, "col": 16},
        )

        explicit = AddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 4, "col": 16},
            hw_temp_stride={"row": 16, "col": 1},
        )

        assert contiguous.get_addr_list() == explicit.get_addr_list()

    def test_contiguous_3d(self):
        """Test 3D contiguous mapping with auto-computed strides."""
        contiguous = ContiguousAddressMapping(
            base=0,
            hw_temp_einops="a b c",
            hw_temp_dim={"a": 4, "b": 3, "c": 2},
        )

        # Expected strides: a=3*2=6, b=2*1=2, c=1
        assert contiguous.hw_temp_stride == {"a": 6, "b": 2, "c": 1}

        addrs = contiguous.get_addr_list()
        assert len(addrs) == 4 * 3 * 2

        # First address is 0
        assert addrs[0] == 0
        # Last address is (3*6 + 2*2 + 1*1) = 18+4+1 = 23
        assert addrs[-1] == 23

    def test_contiguous_with_base(self):
        """Test contiguous mapping with base offset."""
        contiguous = ContiguousAddressMapping(
            base=0x1000,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 4, "col": 16},
        )

        addrs = contiguous.get_addr_list()
        assert addrs[0] == 0x1000
        assert addrs[-1] == 0x1000 + (4 * 16 - 1)

    def test_contiguous_1d(self):
        """Test 1D contiguous mapping (stride=1)."""
        contiguous = ContiguousAddressMapping(
            base=0,
            hw_temp_einops="idx",
            hw_temp_dim={"idx": 10},
        )

        assert contiguous.hw_temp_stride == {"idx": 1}
        addrs = contiguous.get_addr_list()
        assert addrs == list(range(10))

    def test_contiguous_is_subclass(self):
        """Test that ContiguousAddressMapping is a subclass of AddressMapping."""
        contiguous = ContiguousAddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 4, "col": 16},
        )
        assert isinstance(contiguous, AddressMapping)

    def test_contiguous_returns_int_sequence(self):
        """Test that get_addr_list returns IntSequence."""
        contiguous = ContiguousAddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"row": 2, "col": 3},
        )
        addrs = contiguous.get_addr_list()
        assert isinstance(addrs, IntSequence)

    def test_contiguous_addresses_are_sequential(self):
        """Test that contiguous addresses cover 0..N-1 range."""
        contiguous = ContiguousAddressMapping(
            base=0,
            hw_temp_einops="a b c",
            hw_temp_dim={"a": 3, "b": 4, "c": 2},
        )
        addrs = contiguous.get_addr_list()
        assert sorted(addrs) == list(range(3 * 4 * 2))

    def test_address_mapping_einops_key_mismatch(self):
        """Test that mismatched keys between hw_temp_einops and hw_temp_dim raises."""
        with pytest.raises(AssertionError):
            AddressMapping(
                base=0,
                hw_temp_einops="row col",
                hw_temp_dim={"row": 4, "xxx": 16},
                hw_temp_stride={"row": 16, "col": 1},
            )

    def test_address_mapping_einops_stride_key_mismatch(self):
        """Test that mismatched keys between hw_temp_einops and hw_temp_stride raises."""
        with pytest.raises(AssertionError):
            AddressMapping(
                base=0,
                hw_temp_einops="row col",
                hw_temp_dim={"row": 4, "col": 16},
                hw_temp_stride={"row": 16, "xxx": 1},
            )

    def test_address_mapping_unordered_dicts(self):
        """Test that dicts passed in different key order still work correctly."""
        # Pass dicts in reversed key order vs hw_temp_einops
        mapping = AddressMapping(
            base=0,
            hw_temp_einops="row col",
            hw_temp_dim={"col": 16, "row": 4},
            hw_temp_stride={"col": 1, "row": 16},
        )
        addrs = mapping.get_addr_list()
        assert len(addrs) == 64
        # row=0,col=0 -> 0; row=0,col=1 -> 1; row=1,col=0 -> 16
        assert addrs[0] == 0
        assert addrs[1] == 1
        assert addrs[16] == 16


class TestTileMapping:
    """Tests for TileMapping class."""

    def test_tile_mapping_simple_3d(self, sample_tensor_chw):
        """Test simple CHW to 2D matrix transformation."""
        tensor = sample_tensor_chw  # Shape: (3, 8, 8)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",  # 2D: c=temporal, (h w)=spatial
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
        )

        # Convert tensor using the mapping
        seq = tensor_to_cocotb_seq(tensor, mapping)
        recovered = cocotb_seq_to_tensor(seq, mapping)

        # Verify roundtrip
        assert torch.equal(tensor, recovered)

    def test_tile_mapping_grouped_dimensions(self):
        """Test with grouped dimensions like (ht hs) (wt ws) ct."""
        torch.manual_seed(42)
        # Create tensor: (ht*hs=8, wt*ws=8, ct=3)
        tensor = torch.randn(8, 8, 3, dtype=torch.float32)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="(ht hs) (wt ws) ct",
            hw_einops="(ht wt ct) (hs ws)",
            hw_temp_dim={"ht": 2, "wt": 2, "ct": 3},
            hw_spat_dim={"hs": 4, "ws": 4},
        )

        # Convert and recover
        seq = tensor_to_cocotb_seq(tensor, mapping)
        recovered = cocotb_seq_to_tensor(seq, mapping)

        # Verify roundtrip
        assert torch.equal(tensor, recovered)

    def test_tile_mapping_with_address_mapping(self, sample_tensor_chw):
        """Test TileMapping with a separate AddressMapping."""
        tensor = sample_tensor_chw  # Shape: (3, 8, 8)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
        )

        addr_mapping = AddressMapping(
            base=0x1000,
            hw_temp_einops="c",
            hw_temp_dim={"c": 3},
            hw_temp_stride={"c": 64},
        )

        # Check that address mapping works correctly
        assert addr_mapping.base == 0x1000

        # Get address list and verify stride
        addrs = addr_mapping.get_addr_list()
        assert len(addrs) == 3  # One address per channel

        # Each channel should be 64 addresses apart
        assert addrs[1] - addrs[0] == 64
        assert addrs[2] - addrs[1] == 64

    def test_tile_mapping_num_attribute(self):
        """Test that num attribute is calculated correctly."""
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
        )

        # num should be product of hw_spat_dim values
        expected_num = 8 * 8
        assert mapping.num == expected_num

    def test_tile_mapping_formulas(self):
        """Test that sw_to_hw and hw_to_sw formulas are generated correctly."""
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
        )

        assert mapping.sw_to_hw_formula == "c h w -> c (h w)"
        assert mapping.hw_to_sw_formula == "c (h w) -> c h w"

    def test_tile_mapping_to_hw_to_sw(self, sample_tensor_chw):
        """Test to_hw and to_sw methods."""
        tensor = sample_tensor_chw  # Shape: (3, 8, 8)

        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
        )

        addr_mapping = AddressMapping(
            base=0x1000,
            hw_temp_einops="c",
            hw_temp_dim={"c": 3},
            hw_temp_stride={"c": 64},
        )

        # to_hw: convert tensor to values only
        values = mapping.to_hw(tensor)
        assert len(values) == 3
        assert isinstance(values, IntSequence)

        # Verify addresses separately
        addrs = addr_mapping.get_addr_list()
        assert len(addrs) == 3
        assert addrs == [0x1000, 0x1000 + 64, 0x1000 + 128]

        # to_sw: convert back (values only)
        tensor_restored = mapping.to_sw(values)
        assert torch.equal(tensor, tensor_restored)

    def test_tile_mapping_invalid_hw_format(self):
        """Test that invalid hw_einops format is rejected."""
        with pytest.raises(AssertionError, match="2D format"):
            TileMapping(
                dtype=torch.float32,
                sw_einops="c h w",
                hw_einops="(c h w)",  # Invalid: only 1 group!
                hw_temp_dim={"c": 3},
                hw_spat_dim={"h": 8, "w": 8},
            )

    def test_tile_mapping_to_hw_returns_int_sequence(self):
        """Test that to_hw() returns IntSequence type."""
        mapping = TileMapping(
            dtype=torch.float32,
            sw_einops="c h w",
            hw_einops="c (h w)",
            hw_temp_dim={"c": 3},
            hw_spat_dim={"h": 8, "w": 8},
        )

        tensor = torch.randn(3, 8, 8)
        values = mapping.to_hw(tensor)
        assert isinstance(values, IntSequence)
        assert len(values) == 3
