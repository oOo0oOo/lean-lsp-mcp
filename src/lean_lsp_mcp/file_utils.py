import os
from typing import Optional, Dict

from mcp.server.fastmcp import Context
from leanclient import LeanLSPClient


def get_relative_file_path(lean_project_path: str, file_path: str) -> Optional[str]:
    """Convert path relative to project path.

    Args:
        lean_project_path (str): Path to the Lean project root.
        file_path (str): File path.

    Returns:
        str: Relative file path.
    """
    # Check if absolute path
    if os.path.exists(file_path):
        return os.path.relpath(file_path, lean_project_path)

    # Check if relative to project path
    path = os.path.join(lean_project_path, file_path)
    if os.path.exists(path):
        return os.path.relpath(path, lean_project_path)

    # Check if relative to CWD
    cwd = os.getcwd().strip()  # Strip necessary?
    path = os.path.join(cwd, file_path)
    if os.path.exists(path):
        return os.path.relpath(path, lean_project_path)

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
    abs_path = os.path.join(
        ctx.request_context.lifespan_context.lean_project_path, rel_path
    )
    file_content = get_file_contents(abs_path)
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
    except Exception as e:
        pass
    return file_content
