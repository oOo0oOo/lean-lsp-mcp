import os

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import StdoutToStderr


logger = get_logger(__name__)


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
    """
    lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    if lean_project_path is None:
        raise ValueError("lean project path is not set.")

    # Check if already correct client
    client: LeanLSPClient | None = ctx.request_context.lifespan_context.client

    if client is not None:
        if client.project_path == lean_project_path:
            return
        client.close()
        ctx.request_context.lifespan_context.file_content_hashes.clear()

    with StdoutToStderr():
        try:
            client = LeanLSPClient(
                lean_project_path, initial_build=True, print_warnings=False
            )
            logger.info(f"Connected to Lean language server at {lean_project_path}")
        except Exception as e:
            client = LeanLSPClient(
                lean_project_path, initial_build=False, print_warnings=False
            )
            logger.error(f"Could not do initial build, error: {e}")
    ctx.request_context.lifespan_context.client = client


def valid_lean_project_path(path: str) -> bool:
    """Check if the given path is a valid Lean project path (contains a lean-toolchain file).

    Args:
        path (str): Absolute path to check.

    Returns:
        bool: True if valid Lean project path, False otherwise.
    """
    if not os.path.exists(path):
        return False
    return os.path.isfile(os.path.join(path, "lean-toolchain"))


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Check if the current LSP client is already set up and correct for this file. Otherwise, set it up.

    Args:
        ctx (Context): Context object.
        file_path (str): Absolute path to the Lean file.

    Returns:
        str: Relative file path if the client is set up correctly, otherwise None.
    """
    # Check if the file_path works for the current lean_project_path.
    lean_project_path = ctx.request_context.lifespan_context.lean_project_path
    if lean_project_path is not None:
        rel_path = get_relative_file_path(lean_project_path, file_path)
        if rel_path is not None:
            startup_client(ctx)
            return rel_path

    # Try to find the new correct project path by checking all directories in file_path.
    file_dir = os.path.dirname(file_path)
    rel_path = None
    while file_dir:
        if valid_lean_project_path(file_dir):
            lean_project_path = file_dir
            rel_path = get_relative_file_path(lean_project_path, file_path)
            if rel_path is not None:
                ctx.request_context.lifespan_context.lean_project_path = (
                    lean_project_path
                )
                startup_client(ctx)
                break
        # Move up one directory
        file_dir = os.path.dirname(file_dir)

    return rel_path
