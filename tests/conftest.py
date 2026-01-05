"""Pytest configuration and shared fixtures for torchbit tests."""

import pytest
import torch
import numpy as np
import torchbit
from torchbit.tools.buffer import TwoPortBuffer
from torchbit.tools.mapping import TileMapping, AddressMapping


def pytest_configure(config):
    """Display which torchbit version is being used at test session start."""
    torchbit_path = torchbit.__file__
    if "torchbit-test" in torchbit_path:
        print(f"\n[torchbit] Using DEVELOPMENT version")
        print(f"          {torchbit_path}")
    elif "site-packages" in torchbit_path:
        print(f"\n[torchbit] Using INSTALLED version (from conda env)")
        print(f"          {torchbit_path}")
        print(f"          To use dev version: export PYTHONPATH=/home/sdu/torchbit_project/torchbit-test:$PYTHONPATH")
    else:
        print(f"\n[torchbit] Using: {torchbit_path}")


@pytest.fixture
def sample_tensor_2d():
    """Create a sample 2D tensor for testing."""
    return torch.randn(4, 4, dtype=torch.float32)


@pytest.fixture
def sample_tensor_chw():
    """Create a sample CHW tensor (channels=3, height=8, width=8)."""
    torch.manual_seed(42)
    return torch.randn(3, 8, 8, dtype=torch.float32)


@pytest.fixture
def sample_tensor_nchw():
    """Create a sample NCHW tensor (batch=2, channels=3, height=4, width=4)."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 4, 4, dtype=torch.float32)


@pytest.fixture
def buffer_32x1024():
    """Create a TwoPortBuffer with 32-bit width and 1024 depth."""
    return TwoPortBuffer(width=32, depth=1024)


@pytest.fixture
def buffer_128x512():
    """Create a TwoPortBuffer with 128-bit width and 512 depth."""
    return TwoPortBuffer(width=128, depth=512)


@pytest.fixture
def buffer_256x256():
    """Create a TwoPortBuffer with 256-bit width and 256 depth."""
    return TwoPortBuffer(width=256, depth=256)


@pytest.fixture(params=[torch.float32, torch.int8, torch.uint8, torch.float16])
def dtype(request):
    """Parametrized fixture for common dtypes."""
    return request.param


@pytest.fixture
def simple_mapping_chw():
    """Create a simple TileMapping for CHW to linear memory."""
    return TileMapping(
        dtype=torch.float32,
        sw_einops="c h w -> c h w",
        hw_einops="c h w -> (c h w)",
        hw_temp_dim={"c": 3},
        hw_spat_dim={"h": 8, "w": 8},
        base_addr=0,
        strides=None,
    )


@pytest.fixture
def mapping_nhwc():
    """Create a TileMapping for NHWC (channel-last) layout."""
    return TileMapping(
        dtype=torch.float32,
        sw_einops="c h w -> c h w",
        hw_einops="c h w -> (h w c)",
        hw_temp_dim={"c": 3},
        hw_spat_dim={"h": 8, "w": 8},
        base_addr=0,
        strides=None,
    )


@pytest.fixture
def mapping_tiled():
    """Create a TileMapping for 2D tiled layout."""
    return TileMapping(
        dtype=torch.float32,
        sw_einops="c h w -> c h w",
        hw_einops="c (h th) (w tw) -> (th tw c) th tw",
        hw_temp_dim={"c": 3},
        hw_spat_dim={"th": 2, "tw": 2},
        base_addr=0,
        strides=None,
    )


@pytest.fixture
def mapping_strided():
    """Create a TileMapping with address strides."""
    return TileMapping(
        dtype=torch.float32,
        sw_einops="c h w -> c h w",
        hw_einops="c h w -> (c h w)",
        hw_temp_dim={"c": 3},
        hw_spat_dim={"h": 8, "w": 8},
        base_addr=0x1000,
        strides={"c": 64},
    )
