"""Tests that torchbit works without cocotb installed.

Uses subprocess to run Python with cocotb temporarily hidden from the import system.
"""
import subprocess
import sys
import pytest


HIDE_COCOTB_PREAMBLE = (
    "import sys; "
    # Block cocotb and cocotb_tools from being imported
    "sys.modules['cocotb'] = None; "
    "sys.modules['cocotb.types'] = None; "
    "sys.modules['cocotb.triggers'] = None; "
    "sys.modules['cocotb.utils'] = None; "
    "sys.modules['cocotb.handle'] = None; "
    "sys.modules['cocotb_tools'] = None; "
    "sys.modules['cocotb_tools.runner'] = None; "
)


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    """Run a Python snippet in a subprocess with cocotb hidden."""
    full_code = HIDE_COCOTB_PREAMBLE + code
    return subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestNoCocotbImport:
    """Verify that core torchbit modules import without cocotb."""

    def test_import_torchbit(self):
        result = _run_snippet("import torchbit; print('ok')")
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_core_vector(self):
        result = _run_snippet(
            "from torchbit.core import Vector, VectorSequence, LogicSequence; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_tiling(self):
        result = _run_snippet(
            "from torchbit.tiling import TileMapping, AddressMapping; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_bit_struct(self):
        result = _run_snippet(
            "from torchbit.core.bit_struct import BitStruct, BitField; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_buffer(self):
        result = _run_snippet(
            "from torchbit.tools.buffer import Buffer; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_buffer_works_without_cocotb(self):
        result = _run_snippet(
            "from torchbit.tools.buffer import Buffer; "
            "buf = Buffer(width=32, depth=1024); "
            "buf.write(0, 42); "
            "assert buf.read(0) == 42; "
            "print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_vector_from_logic_int_works(self):
        """Vector.from_logic() with int value should work without cocotb."""
        result = _run_snippet(
            "import torch; "
            "from torchbit.core import Vector; "
            "vec = Vector.from_logic(0x42480000, 1, torch.float32); "
            "print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_vector_to_logic_works(self):
        """Vector.to_logic() should work without cocotb."""
        result = _run_snippet(
            "import torch; "
            "from torchbit.core import Vector; "
            "vec = Vector.from_array(torch.tensor([1.0, 2.0], dtype=torch.float32)); "
            "val = vec.to_logic(); "
            "print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_debug(self):
        result = _run_snippet(
            "from torchbit import debug; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_utils(self):
        result = _run_snippet(
            "from torchbit import utils; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_import_runner_config(self):
        result = _run_snippet(
            "from torchbit.runner.config import FileConfig, BuildConfig; print('ok')"
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout
