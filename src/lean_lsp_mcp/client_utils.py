import os
from pathlib import Path
from threading import Lock

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import (
    LeanPathPolicy,
    build_lean_path_policy,
    require_lean_project_path,
    resolve_input_path,
    valid_lean_project_path,
)
from lean_lsp_mcp.utils import OutputCapture


logger = get_logger(__name__)
CLIENT_LOCK = Lock()
_ACTIVE_TRANSPORT_ENV = "LEAN_LSP_MCP_ACTIVE_TRANSPORT"


def _active_transport(ctx: Context | None = None) -> str:
    if ctx is not None:
        lifespan = ctx.request_context.lifespan_context
        transport = getattr(lifespan, "active_transport", None)
        if isinstance(transport, str) and transport:
            return transport
    return os.environ.get(_ACTIVE_TRANSPORT_ENV, "stdio").strip().lower() or "stdio"


def _project_switching_allowed(ctx: Context | None = None) -> bool:
    if ctx is not None:
        lifespan = ctx.request_context.lifespan_context
        explicit = getattr(lifespan, "project_switching_allowed", None)
        if explicit is not None:
            return bool(explicit)
    return _active_transport(ctx) == "stdio"


def bind_lean_project_path(ctx: Context, project_path: Path | str) -> Path:
    lifespan = ctx.request_context.lifespan_context
    resolved_project = require_lean_project_path(project_path)
    current_root: Path | None = getattr(lifespan, "lean_project_path", None)
    if current_root is not None:
        current_root = current_root.resolve(strict=False)

    if (
        current_root is not None
        and current_root != resolved_project
        and not _project_switching_allowed(ctx)
    ):
        raise ValueError(
            f"Project switching is disabled for `{_active_transport(ctx)}` transport. "
            "Restart the server with LEAN_PROJECT_PATH set to the desired Lean project root."
        )

    if current_root != resolved_project:
        client: LeanLSPClient | None = getattr(lifespan, "client", None)
        if client is not None and client.project_path != resolved_project:
            try:
                client.close()
            except Exception:
                logger.exception("Lean client close failed during project switch")
            finally:
                lifespan.client = None
        lifespan.lean_project_path = resolved_project

    return resolved_project


def get_path_policy(ctx: Context, project_path: Path | None = None) -> LeanPathPolicy:
    lifespan = ctx.request_context.lifespan_context
    root = project_path or getattr(lifespan, "lean_project_path", None)
    if root is None:
        raise ValueError("lean project path is not set.")
    return build_lean_path_policy(root)


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
    """
    with CLIENT_LOCK:
        configured_root = ctx.request_context.lifespan_context.lean_project_path
        if configured_root is None:
            raise ValueError("lean project path is not set.")
        try:
            lean_project_path = bind_lean_project_path(ctx, configured_root)
        except ValueError as exc:
            raise ValueError("lean project path is not set.") from exc

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
        try:
            with OutputCapture() as output:
                client = LeanLSPClient(
                    lean_project_path,
                    initial_build=False,
                    prevent_cache_get=prevent_cache,
                )
                logger.info(f"Connected to Lean language server at {lean_project_path}")
            build_output = output.get_output()
            if build_output:
                logger.debug(f"Build output: {build_output}")
        except Exception as e:
            logger.exception("Failed to start Lean LSP client")
            raise ValueError(
                f"Failed to start Lean language server at '{lean_project_path}': {e}"
            ) from e
        ctx.request_context.lifespan_context.client = client


def resolve_file_path(
    ctx: Context, file_path: str, *, require_exists: bool = True
) -> Path:
    """Resolve a file path with support for project-root-relative inputs.

    If `LEAN_PROJECT_PATH` is known and `file_path` is relative, it is resolved
    relative to that project root.
    """
    lifespan = ctx.request_context.lifespan_context
    project_root: Path | None = getattr(lifespan, "lean_project_path", None)
    return resolve_input_path(
        file_path, project_root=project_root, require_exists=require_exists
    )


def _pick_project_root(file_path: Path, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None

    for candidate in candidates:
        try:
            relative = file_path.relative_to(candidate)
        except ValueError:
            continue
        if relative.parts[:2] == (".lake", "packages"):
            return candidate
    return candidates[0]


def _cacheable_project_dirs(project_path: Path, cache_dirs: list[str]) -> list[str]:
    cacheable: list[str] = []
    for directory in cache_dirs:
        directory_path = Path(directory)
        try:
            directory_path.relative_to(project_path)
        except ValueError:
            continue
        cacheable.append(directory)
    return cacheable


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

    if ctx is not None:
        resolved_input = resolve_file_path(ctx, file_path, require_exists=False)
    else:
        resolved_input = resolve_input_path(file_path, require_exists=False)

    start_dir = resolved_input if resolved_input.is_dir() else resolved_input.parent
    start_dir = start_dir.resolve(strict=False)
    file_dir = str(start_dir)

    def cache_project_path(project_path: Path, cache_dirs: list[str]) -> Path:
        if ctx:
            bound_project = bind_lean_project_path(ctx, project_path)
            cache_targets = _cacheable_project_dirs(bound_project, cache_dirs)
            for directory in set(cache_targets + [str(bound_project)]):
                if directory:
                    lifespan.project_cache[directory] = bound_project
            return bound_project
        return project_path

    # Fast path: current project already valid for this file
    if ctx and lifespan.lean_project_path:
        try:
            current_policy = build_lean_path_policy(lifespan.lean_project_path)
        except ValueError:
            current_policy = None
        if current_policy is not None and current_policy.contains(resolved_input):
            return cache_project_path(lifespan.lean_project_path, [file_dir])

    # Walk up directory tree using cache and lean-toolchain detection
    current_dir = start_dir
    cache_dirs: list[str] = []
    candidates: list[Path] = []
    while True:
        current_dir_str = str(current_dir)
        cache_dirs.append(current_dir_str)

        if ctx:
            cached_root = lifespan.project_cache.get(current_dir_str)

            if cached_root:
                try:
                    cached_policy = build_lean_path_policy(Path(cached_root))
                except ValueError:
                    lifespan.project_cache[current_dir_str] = ""
                else:
                    if cached_policy.contains(resolved_input):
                        return cache_project_path(Path(cached_root), cache_dirs)
                    lifespan.project_cache[current_dir_str] = ""
        if valid_lean_project_path(current_dir):
            candidates.append(current_dir.resolve(strict=True))
        elif ctx:
            lifespan.project_cache[current_dir_str] = ""  # Mark as checked

        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent

    if chosen_root := _pick_project_root(resolved_input, candidates):
        return cache_project_path(chosen_root, cache_dirs)

    return None


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Ensure the LSP client matches the file's Lean project and return its relative path."""
    try:
        resolved_file = str(resolve_file_path(ctx, file_path))
    except (FileNotFoundError, OSError):
        return None

    project_path = infer_project_path(resolved_file, ctx=ctx)
    if project_path is None:
        return None

    try:
        policy = build_lean_path_policy(project_path)
    except ValueError:
        return None
    if not policy.contains(resolved_file):
        return None

    startup_client(ctx)
    return policy.client_relative_path(resolved_file)
