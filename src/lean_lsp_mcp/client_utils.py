import os
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
    """Ensure the LSP client matches the file's Lean project and return its relative path."""

    lifespan = ctx.request_context.lifespan_context
    project_cache = getattr(lifespan, "project_cache", {})
    if not hasattr(lifespan, "project_cache"):
        lifespan.project_cache = project_cache

    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    def activate_project(project_path: Path, cache_dirs: list[str]) -> str | None:
        project_path_obj = project_path
        rel = get_relative_file_path(project_path_obj, file_path)
        if rel is None:
            return None

        project_path_obj = project_path_obj.resolve()
        lifespan.lean_project_path = project_path_obj

        cache_targets: list[str] = []
        for directory in cache_dirs + [str(project_path_obj)]:
            if directory and directory not in cache_targets:
                cache_targets.append(directory)

        for directory in cache_targets:
            project_cache[directory] = project_path_obj

        startup_client(ctx)
        return rel

    # Fast path: current Lean project already valid for this file
    if lifespan.lean_project_path is not None:
        rel_path = activate_project(lifespan.lean_project_path, [file_dir])
        if rel_path is not None:
            return rel_path

    # Walk up from file directory to root, using cache hits or lean-toolchain
    prev_dir = None
    current_dir = file_dir
    while current_dir and current_dir != prev_dir:
        cached_root = project_cache.get(current_dir)
        if cached_root:
            rel_path = activate_project(Path(cached_root), [current_dir])
            if rel_path is not None:
                return rel_path
        elif valid_lean_project_path(current_dir):
            rel_path = activate_project(Path(current_dir), [current_dir])
            if rel_path is not None:
                return rel_path
        else:
            project_cache[current_dir] = ""
        prev_dir = current_dir
        current_dir = os.path.dirname(current_dir)

    return None
