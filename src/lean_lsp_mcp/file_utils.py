import os
from typing import Optional, Dict
from pathlib import Path

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient


logger = get_logger(__name__)


def get_relative_file_path(lean_project_path: Path, file_path: str) -> Optional[str]:
    """Convert path relative to project path.

    Args:
        lean_project_path (Path): Path to the Lean project root.
        file_path (str): File path.

    Returns:
        str: Relative file path.
    """
    file_path_obj = Path(file_path)

    # Check if absolute path
    if file_path_obj.is_absolute() and file_path_obj.exists():
        try:
            return str(file_path_obj.relative_to(lean_project_path))
        except ValueError:
            # File is not in this project
            return None

    # Check if relative to project path
    path = lean_project_path / file_path
    if path.exists():
        return str(path.relative_to(lean_project_path))

    # Check if relative to CWD
    cwd = Path.cwd()
    path = cwd / file_path
    if path.exists():
        try:
            return str(path.relative_to(lean_project_path))
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


def update_file(ctx: Context, rel_path: str) -> str:
    """Update the file contents in the context.
    Args:
        ctx (Context): Context object.
        rel_path (str): Relative file path.

    Returns:
        str: Updated file contents.
    """
    # Get file contents and hash
    abs_path = ctx.request_context.lifespan_context.lean_project_path / rel_path
    file_content = get_file_contents(str(abs_path))
    hashed_file = hash(file_content)

    # Check if file_contents have changed
    file_content_hashes: Dict[str, str] = (
        ctx.request_context.lifespan_context.file_content_hashes
    )
    if rel_path not in file_content_hashes:
        file_content_hashes[rel_path] = hashed_file
        return file_content

    elif hashed_file == file_content_hashes[rel_path]:
        return file_content

    # Update file_contents
    file_content_hashes[rel_path] = hashed_file

    # Reload file in LSP
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    try:
        client.close_files([rel_path])
    except FileNotFoundError as e:
        logger.warning(
            f"Attempted to close file {rel_path} that wasn't open in LSP client: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error closing file {rel_path}: {type(e).__name__}: {e}"
        )
    return file_content
