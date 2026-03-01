import builtins
import os
from pathlib import Path, PurePosixPath
from threading import Lock

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import OutputCapture


logger = get_logger(__name__)
CLIENT_LOCK = Lock()


def _patch_file_encoding() -> None:
    """Fix leanclient Windows bug: file reads use default encoding (cp1252)
    instead of UTF-8, causing codec errors on Lean files with math symbols.

    Monkeypatch the `open` builtin within leanclient.file_manager to default
    to UTF-8 for text-mode reads.
    """
    if os.name != "nt":
        return
    try:
        import leanclient.file_manager as _fm
    except ImportError:
        return
    if getattr(_fm, "_open_patched", False):
        return  # Already patched

    _original_open = builtins.open

    def _utf8_open(file, mode="r", *args, **kwargs):
        if "b" not in str(mode) and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _original_open(file, mode, *args, **kwargs)

    _fm.open = _utf8_open
    _fm._open_patched = True


_patch_file_encoding()


def _patch_uri_to_local(client: LeanLSPClient) -> None:
    """Fix leanclient Windows bug: _uri_to_local returns backslashes but
    open_files stores keys with forward slashes, causing dict lookup failures.

    Monkeypatch to always return forward-slash paths.
    """
    if os.name != "nt":
        return
    if not hasattr(client, "_uri_to_local"):
        return
    original = client._uri_to_local

    def _posix_uri_to_local(uri: str) -> str:
        result = original(uri)
        return result.replace("\\", "/")

    client._uri_to_local = _posix_uri_to_local


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
            try:
                client.close()
            except Exception:
                logger.exception("Lean client close failed")

        # Need to create a new client
        # In test environments, prevent repeated cache downloads
        prevent_cache = bool(os.environ.get("LEAN_LSP_TEST_MODE"))
        with OutputCapture() as output:
            client = LeanLSPClient(
                lean_project_path, initial_build=False, prevent_cache_get=prevent_cache
            )
            logger.info(f"Connected to Lean language server at {lean_project_path}")
        build_output = output.get_output()
        if build_output:
            logger.debug(f"Build output: {build_output}")
        _patch_uri_to_local(client)
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

    # Walk up directory tree using cache and lean-toolchain detection.
    # Collect ALL lean-toolchain hits so we can prefer the outermost project
    # (inner ones are typically vendored dependencies that can't serve files).
    current_dir = file_dir
    candidates: list[str] = []
    while current_dir and current_dir != os.path.dirname(current_dir):
        if ctx:
            cached_root = lifespan.project_cache.get(current_dir)
            if cached_root:
                candidates.append(cached_root)
                current_dir = os.path.dirname(current_dir)
                continue
        if valid_lean_project_path(current_dir):
            candidates.append(current_dir)
        elif ctx:
            lifespan.project_cache[current_dir] = ""  # Mark as checked
        current_dir = os.path.dirname(current_dir)

    # Prefer the outermost project (last found), since inner projects are
    # typically vendored sub-projects whose lake serve may not work.
    for candidate_dir in reversed(candidates):
        if result := set_project_path(Path(candidate_dir), [file_dir]):
            return result

    return None


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Ensure the LSP client matches the file's Lean project and return its relative path."""
    project_path = infer_project_path(file_path, ctx=ctx)
    if project_path is None:
        return None

    startup_client(ctx)
    return get_relative_file_path(project_path, file_path)
