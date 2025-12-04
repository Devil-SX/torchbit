from pathlib import Path

def read_filelist(filelist_path: Path, base_path: Path = None) -> list[Path]:
    """Reads a filelist and returns a list of file paths.

    Args:
        filelist_path (Path): Path to the filelist.
        base_path (Path): Optional base path to resolve file paths against.
                          If None, defaults to filelist_path.parent.
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