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

# ---------------------------------------------------------------------------
# Shared LSP client singleton.
#
# With ``streamable-http`` transport every MCP session gets its own
# ``app_lifespan`` invocation.  The LSP client (which spawns ``lake serve``)
# is expensive to create and consumes significant RAM, so we share a single
# instance across all sessions — same pattern used for loogle.
#
# For ``stdio`` transport there is only one session, so this is a no-op.
# ---------------------------------------------------------------------------
_shared_client: LeanLSPClient | None = None
_shared_client_project: Path | None = None
_build_in_progress: bool = False


def _get_or_create_shared_client(lean_project_path: Path) -> LeanLSPClient:
    """Return the shared LSP client, creating it if needed.

    Must be called while holding CLIENT_LOCK.
    """
    global _shared_client, _shared_client_project

    if _build_in_progress:
        raise ValueError(
            "A project build is in progress. Retry after the build completes."
        )

    if _shared_client is not None:
        if _shared_client_project == lean_project_path:
            return _shared_client
        # Project changed — close old client
        try:
            _shared_client.close()
        except Exception:
            logger.exception("Shared Lean client close failed")
        _shared_client = None
        _shared_client_project = None

    prevent_cache = bool(os.environ.get("LEAN_LSP_TEST_MODE"))
    try:
        with OutputCapture() as output:
            client = LeanLSPClient(
                lean_project_path,
                initial_build=False,
                prevent_cache_get=prevent_cache,
            )
            logger.info(f"Shared LSP client connected at {lean_project_path}")
        build_output = output.get_output()
        if build_output:
            logger.debug(f"Build output: {build_output}")
    except Exception as e:
        logger.exception("Failed to start shared Lean LSP client")
        raise ValueError(
            f"Failed to start Lean language server at '{lean_project_path}': {e}"
        ) from e

    _shared_client = client
    _shared_client_project = lean_project_path
    return client


def set_build_in_progress(value: bool) -> None:
    """Mark whether a build is in progress.

    While True, ``_get_or_create_shared_client`` will refuse to spawn a new
    ``lake serve`` (the oleans are being rewritten and any new LSP would give
    wrong results).  Must be called while holding CLIENT_LOCK.
    """
    assert CLIENT_LOCK.locked(), "set_build_in_progress requires CLIENT_LOCK"
    global _build_in_progress
    _build_in_progress = value


def replace_shared_client(
    client: LeanLSPClient | None, project_path: Path | None
) -> None:
    """Replace the shared client (e.g. after ``lean_build`` restarts the LSP).

    Must be called while holding CLIENT_LOCK.
    """
    assert CLIENT_LOCK.locked(), "replace_shared_client requires CLIENT_LOCK"
    global _shared_client, _shared_client_project
    _shared_client = client
    _shared_client_project = project_path


def close_shared_client() -> None:
    """Close the shared client (for clean shutdown)."""
    global _shared_client, _shared_client_project
    with CLIENT_LOCK:
        if _shared_client is not None:
            try:
                _shared_client.close()
            except Exception:
                logger.exception("Shared Lean client close failed during shutdown")
            _shared_client = None
            _shared_client_project = None


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Uses a shared singleton so that multiple MCP sessions (e.g. from
    sequential ``claude -p`` calls against an HTTP server) reuse the same
    ``lake serve`` process instead of spawning a new one each time.

    Args:
        ctx (Context): Context object.
    """
    with CLIENT_LOCK:
        lean_project_path = ctx.request_context.lifespan_context.lean_project_path
        if lean_project_path is None:
            raise ValueError("lean project path is not set.")

        client = _get_or_create_shared_client(lean_project_path)
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


def infer_project_path(file_path: str, ctx: Context | None = None) -> Path | None:
    """Infer and cache the Lean project path for a file WITHOUT starting the client.

    Walks up the directory tree to find a lean-toolchain file, caches the result.
    Sets ctx.request_context.lifespan_context.lean_project_path if found.
    If ctx is None, only returns the resolved project path.

    Side effects when path changes when ctx is not None:
    - Next LSP tool will restart the client for the new project
    - File content hashes will be cleared

    Args:
        file_path (str): Absolute or relative path to a Lean file
        ctx (Context): Context object, or None to only infer and return.

    Returns:
        Path | None: The resolved project path if found, None otherwise
    """
    if ctx:
        lifespan = ctx.request_context.lifespan_context
        if not hasattr(lifespan, "project_cache"):
            lifespan.project_cache = {}

    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    def set_project_path(project_path: Path, cache_dirs: list[str]) -> Path | None:
        """Validate file is in project, set path, update cache."""
        if get_relative_file_path(project_path, file_path) is None:
            return None

        project_path = project_path.resolve()
        if ctx:
            lifespan.lean_project_path = project_path

            # Update all relevant directories in cache
            for directory in set(cache_dirs + [str(project_path)]):
                if directory:
                    lifespan.project_cache[directory] = project_path

        return project_path

    # Fast path: current project already valid for this file
    if (
        ctx
        and lifespan.lean_project_path
        and set_project_path(lifespan.lean_project_path, [file_dir])
    ):
        return lifespan.lean_project_path

    # Walk up directory tree using cache and lean-toolchain detection
    current_dir = file_dir
    while current_dir and current_dir != os.path.dirname(current_dir):
        if ctx:
            cached_root = lifespan.project_cache.get(current_dir)

            if cached_root:
                if result := set_project_path(Path(cached_root), [current_dir]):
                    return result
        if valid_lean_project_path(current_dir):
            if result := set_project_path(Path(current_dir), [current_dir]):
                return result
        elif ctx:
            lifespan.project_cache[current_dir] = ""  # Mark as checked

        current_dir = os.path.dirname(current_dir)

    return None


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Ensure the LSP client matches the file's Lean project and return its relative path."""
    project_path = infer_project_path(file_path, ctx=ctx)
    if project_path is None:
        return None

    startup_client(ctx)
    return get_relative_file_path(project_path, file_path)
