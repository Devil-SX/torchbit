"""
External file format conversion utilities.

Handles conversion of proprietary formats to FST for WAL processing.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class FsdbConverterError(Exception):
    """Error during FSDB conversion."""
    message: str
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


class FsdbConverter:
    """
    Converter for FSDB to FST format.

    Uses fsdb2fst utility from VCS/Verdi installation.
    """

    def __init__(self, fsdb2fst_path: Optional[str] = None):
        """
        Initialize converter.

        Args:
            fsdb2fst_path: Path to fsdb2fst binary. If None, searches PATH.
        """
        self.fsdb2fst_path = fsdb2fst_path or shutil.which("fsdb2fst")
        if self.fsdb2fst_path is None:
            raise FsdbConverterError(
                "fsdb2fst not found. Please ensure VCS/Verdi is installed "
                "and fsdb2fst is in PATH."
            )

    def convert(self, fsdb_path: str, fst_path: str,
                reuse_existing: bool = True) -> str:
        """
        Convert FSDB file to FST format.

        Args:
            fsdb_path: Path to input FSDB file.
            fst_path: Path to output FST file.
            reuse_existing: If True and fst_path exists, skip conversion.

        Returns:
            Path to the FST file.

        Raises:
            FsdbConverterError: If conversion fails.
            FileNotFoundError: If input file doesn't exist.
        """
        fsdb_path = Path(fsdb_path)
        fst_path = Path(fst_path)

        if not fsdb_path.exists():
            raise FileNotFoundError(f"FSDB file not found: {fsdb_path}")

        # Check if we can reuse existing FST
        if reuse_existing and fst_path.exists():
            return str(fst_path)

        # Ensure output directory exists
        fst_path.parent.mkdir(parents=True, exist_ok=True)

        # Run conversion
        cmd = [self.fsdb2fst_path, str(fsdb_path), str(fst_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise FsdbConverterError(
                f"fsdb2fst failed with code {result.returncode}",
                return_code=result.returncode,
                stderr=result.stderr
            )

        return str(fst_path)

    def __repr__(self) -> str:
        return f"FsdbConverter(fsdb2fst={self.fsdb2fst_path!r})"
