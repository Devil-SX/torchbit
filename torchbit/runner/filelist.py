"""
HDL source filelist parsing utilities.

Provides the read_filelist function for parsing Verilog/HDL source
filelists commonly used in large projects.
"""
from pathlib import Path
from typing import List


def read_filelist(filelist_path: Path, base_path: Path = None) -> List[Path]:
    """Read an HDL filelist and return resolved file paths.

    Parses a filelist file (commonly used in Verilog projects) and
    returns a list of resolved absolute file paths. The filelist
    format supports:
    - One file path per line
    - Comments starting with #
    - Empty lines (ignored)

    Paths in the filelist can be relative to the filelist location
    or absolute. The function resolves them relative to base_path.

    Args:
        filelist_path: Path to the filelist file.
        base_path: Optional base path to resolve relative file paths against.
                   If None, defaults to filelist_path.parent.

    Returns:
        List of resolved Path objects for each source file.

    Raises:
        AssertionError: If a resolved file path does not exist.

    Example:
        >>> from pathlib import Path
        >>> from torchbit.runner import read_filelist
        >>>
        >>> # Read filelist with files relative to filelist location
        >>> files = read_filelist("design.f")
        >>> # files[0] might be: PosixPath('/path/to/design/top.sv')
        >>>
        >>> # Read with explicit base path
        >>> files = read_filelist("filelist.txt", base_path=Path("/project"))
        >>>
        >>> # Filelist format example (design.f):
        >>> # /project/src/top.sv
        >>> # /project/src/submodule.sv
        >>> # /project/include/defines.vh
        >>> # # This is a comment
        >>> # /project/src/another_module.sv

    Note:
        The filelist format varies between projects. Common formats include:
        - Plain file paths (one per line)
        - +incdir+ directives for includes
        - -v flag for library files
        - -y flag for library directories

        This function only handles simple file paths. For full support,
        consider using a more sophisticated parser.
    """
    with open(filelist_path, "r") as f:
        lines = f.readlines()

    if base_path is None:
        base_path = Path(filelist_path).parent

    file_paths = []
    for line in lines:
        if line.strip() and not line.startswith("#"):
            file_path = Path(line.strip())
            resolved_file_path = base_path / file_path

            assert resolved_file_path.exists(), f"File not found: {resolved_file_path}"
            file_paths.append(resolved_file_path)

    return file_paths
