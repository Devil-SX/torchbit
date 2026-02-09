"""
HDL source filelist parsing utilities.

Provides the read_filelist function for parsing Verilog/HDL source
filelists commonly used in EDA tool flows (VCS, Xcelium, Verilator, Questa).
"""
from pathlib import Path
from typing import List


def _resolve_path(path_str: str, base_path: Path) -> Path:
    """Resolve a path string against a base path.

    Absolute paths are returned as-is; relative paths are resolved
    against base_path.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return base_path / p


# Directives that take no file argument or are not relevant to source collection.
# These are silently skipped during parsing.
_SKIP_PREFIXES = (
    "+incdir+", "+define+", "+libext+",
    "-v ", "-y ", "-sv", "-timescale",
)


def read_filelist(filelist_path: Path, base_path: Path = None) -> List[Path]:
    """Read an HDL filelist and return resolved source file paths.

    Parses a filelist file following the common EDA command-file conventions
    (IEEE 1800 / Synopsys VCS / Cadence Xcelium / Verilator / Questa).

    Supported syntax:
    - One source file path per line
    - Comments: lines starting with ``#`` or ``//``
    - ``-f <file>``: Include a nested filelist. Relative paths **inside**
      the nested filelist are resolved against the **caller's** base_path.
    - ``-F <file>``: Include a nested filelist. Relative paths **inside**
      the nested filelist are resolved against the **nested filelist's own
      directory**.
    - Known but unsupported directives (``+incdir+``, ``+define+``,
      ``+libext+``, ``-v``, ``-y``, ``-sv``, ``-timescale``) are
      silently skipped.

    Args:
        filelist_path: Path to the filelist file.
        base_path: Base path for resolving relative file paths.
                   Defaults to ``filelist_path.parent``.

    Returns:
        Flat list of resolved :class:`~pathlib.Path` objects.

    Raises:
        AssertionError: If a referenced file or filelist does not exist.

    Example:
        >>> from torchbit.runner import read_filelist
        >>>
        >>> files = read_filelist("design.f")
        >>>
        >>> # Filelist format example (design.f):
        >>> # # Comment
        >>> # // Another comment
        >>> # src/top.sv
        >>> # src/submodule.sv
        >>> # -f  sub_filelist.f      # nested, paths relative to design.f dir
        >>> # -F  lib/lib_filelist.f   # nested, paths relative to lib/
    """
    filelist_path = Path(filelist_path)

    with open(filelist_path, "r") as f:
        lines = f.readlines()

    if base_path is None:
        base_path = filelist_path.parent

    file_paths = []
    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # -f <file>: nested filelist, relative paths inside use caller's base_path
        if line.startswith("-f "):
            nested_path_str = line[3:].strip()
            nested_filelist = _resolve_path(nested_path_str, base_path)
            assert nested_filelist.exists(), f"Filelist not found: {nested_filelist}"
            file_paths.extend(read_filelist(nested_filelist, base_path=base_path))
            continue

        # -F <file>: nested filelist, relative paths inside use filelist's own dir
        if line.startswith("-F "):
            nested_path_str = line[3:].strip()
            nested_filelist = _resolve_path(nested_path_str, base_path)
            assert nested_filelist.exists(), f"Filelist not found: {nested_filelist}"
            file_paths.extend(read_filelist(nested_filelist, base_path=nested_filelist.parent))
            continue

        # Skip known directives that don't contribute source files
        if any(line.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        # Regular source file path
        resolved = _resolve_path(line, base_path)
        assert resolved.exists(), f"File not found: {resolved}"
        file_paths.append(resolved)

    return file_paths
