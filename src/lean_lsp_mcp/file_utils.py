from typing import Optional
from pathlib import Path


def get_relative_file_path(lean_project_path: Path, file_path: str) -> Optional[str]:
    """Convert path relative to project path.

    Args:
        lean_project_path (Path): Path to the Lean project root.
        file_path (str): File path.

    Returns:
        str: Relative file path.
    """
    file_path_obj = Path(file_path)

    # Absolute path under project
    if file_path_obj.is_absolute() and file_path_obj.exists():
        try:
            return str(file_path_obj.relative_to(lean_project_path))
        except ValueError:
            return None

    # Relative to project path
    path = lean_project_path / file_path
    if path.exists():
        return str(path.relative_to(lean_project_path))

    # Relative to CWD, but only if inside project root
    cwd = Path.cwd()
    path = cwd / file_path
    if path.exists():
        try:
            return str(path.resolve().relative_to(lean_project_path))
        except ValueError:
            return None

    return None


def get_file_contents(abs_path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(abs_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(abs_path, "r", encoding=None) as f:
        return f.read()
