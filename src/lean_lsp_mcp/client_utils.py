from pathlib import Path
from threading import Lock

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import OutputCapture


logger = get_logger(__name__)
CLIENT_LOCK = Lock()


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
    """
    with CLIENT_LOCK:
        lean_project_path = ctx.request_context.lifespan_context.lean_project_path
        if lean_project_path is None:
            raise ValueError("lean project path is not set.")

        # Check if already correct client
        client: LeanLSPClient | None = ctx.request_context.lifespan_context.client

        if client is not None:
            # Both are Path objects now, direct comparison works
            if client.project_path == lean_project_path:
                return  # Client already set up correctly - reuse it!
            # Different project path - close old client
            client.close()
            ctx.request_context.lifespan_context.file_content_hashes.clear()

        # Need to create a new client
        with OutputCapture() as output:
            try:
                client = LeanLSPClient(lean_project_path)
                logger.info(f"Connected to Lean language server at {lean_project_path}")
            except Exception as e:
                logger.warning(f"Initial connection failed, trying with build: {e}")
                client = LeanLSPClient(lean_project_path, initial_build=True)
                logger.info(f"Connected with initial build to {lean_project_path}")
        build_output = output.get_output()
        if build_output:
            logger.debug(f"Build output: {build_output}")
        ctx.request_context.lifespan_context.client = client


def valid_lean_project_path(path: Path | str) -> bool:
    """Check if the given path is a valid Lean project path (contains a lean-toolchain file).

    Args:
        path (Path | str): Absolute path to check.

    Returns:
        bool: True if valid Lean project path, False otherwise.
    """
    path_obj = Path(path) if isinstance(path, str) else path
    return (path_obj / "lean-toolchain").is_file()


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

    # Try to find the correct project path by checking all directories in file_path.
    file_path_obj = Path(file_path)
    rel_path = None
    for parent in file_path_obj.parents:
        if valid_lean_project_path(parent):
            lean_project_path = parent
            rel_path = get_relative_file_path(lean_project_path, file_path)
            if rel_path is not None:
                ctx.request_context.lifespan_context.lean_project_path = (
                    lean_project_path
                )
                startup_client(ctx)
                break

    return rel_path
