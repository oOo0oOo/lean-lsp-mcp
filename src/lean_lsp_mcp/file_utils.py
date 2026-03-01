import os
from typing import Optional
from pathlib import Path


def _to_posix(p: str) -> str:
    """Normalize Windows backslashes to forward slashes.

    leanclient stores file keys with the separator used by the caller.
    On Windows, Path.relative_to() returns backslashes, but leanclient's
    _uri_to_local also returns backslashes, causing dict key mismatches
    when the two aren't consistent.  Using forward slashes everywhere
    avoids this.
    """
    return p.replace("\\", "/") if os.name == "nt" else p


def get_relative_file_path(lean_project_path: Path, file_path: str) -> Optional[str]:
    """Convert path relative to project path.

    Args:
        lean_project_path (Path): Path to the Lean project root.
        file_path (str): File path.

    Returns:
        str: Relative file path (always forward slashes).
    """
    file_path_obj = Path(file_path)

    # Absolute path under project
    if file_path_obj.is_absolute() and file_path_obj.exists():
        try:
            return _to_posix(str(file_path_obj.relative_to(lean_project_path)))
        except ValueError:
            return None

    # Relative to project path
    path = lean_project_path / file_path
    if path.exists():
        return _to_posix(str(path.relative_to(lean_project_path)))

    # Relative to CWD, but only if inside project root
    cwd = Path.cwd()
    path = cwd / file_path
    if path.exists():
        try:
            return _to_posix(str(path.resolve().relative_to(lean_project_path)))
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
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
