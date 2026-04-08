import os
from pathlib import Path
from threading import Lock

from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger

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
_shared_clients: dict[Path, LeanLSPClient] = {}
_builds_in_progress: set[Path] = set()


_MAX_SHARED_CLIENTS = 8


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


def _max_opened_files() -> int:
    raw_value = os.environ.get("LEAN_LSP_MAX_OPEN_FILES", "4")
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid LEAN_LSP_MAX_OPEN_FILES=%s, defaulting to 4.", raw_value
        )
        return 4
    if value < 1:
        logger.warning(
            "Invalid LEAN_LSP_MAX_OPEN_FILES=%s, defaulting to 4.", raw_value
        )
        return 4
    return value


def _client_is_alive(client: LeanLSPClient) -> bool:
    process = getattr(client, "process", None)
    if process is None:
        return True

    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            return poll() is None
        except Exception:
            return False

    return getattr(process, "returncode", None) is None


def _close_client(client: LeanLSPClient, message: str) -> None:
    try:
        client.close()
    except Exception:
        logger.exception(message)


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
        lifespan.lean_project_path = resolved_project
        current_client: LeanLSPClient | None = getattr(lifespan, "client", None)
        if (
            current_client is not None
            and getattr(current_client, "project_path", None) != resolved_project
        ):
            lifespan.client = None

    return resolved_project


def get_path_policy(ctx: Context, project_path: Path | None = None) -> LeanPathPolicy:
    lifespan = ctx.request_context.lifespan_context
    root = project_path or getattr(lifespan, "lean_project_path", None)
    if root is None:
        raise ValueError("lean project path is not set.")
    return build_lean_path_policy(root)


def _start_client(project_path: Path) -> LeanLSPClient:
    prevent_cache = bool(os.environ.get("LEAN_LSP_TEST_MODE"))
    try:
        with OutputCapture() as output:
            client = LeanLSPClient(
                project_path,
                initial_build=False,
                prevent_cache_get=prevent_cache,
                max_opened_files=_max_opened_files(),
            )
            logger.info("Shared LSP client connected at %s", project_path)
        build_output = output.get_output()
        if build_output:
            logger.debug("Build output: %s", build_output)
    except Exception as exc:
        logger.exception("Failed to start shared Lean LSP client")
        raise ValueError(
            f"Failed to start Lean language server at '{project_path}': {exc}"
        ) from exc
    return client


def _evict_oldest_client() -> None:
    """Close and remove the oldest shared client to stay within the cap."""
    oldest_key = next(iter(_shared_clients))
    old = _shared_clients.pop(oldest_key)
    _close_client(old, f"Evicted shared client for {oldest_key}")


def _get_or_create_shared_client(lean_project_path: Path) -> LeanLSPClient:
    project_key = lean_project_path.resolve(strict=False)

    if project_key in _builds_in_progress:
        raise ValueError(
            "A project build is in progress. Retry after the build completes."
        )

    client = _shared_clients.get(project_key)
    if client is not None and _client_is_alive(client):
        return client

    if client is not None:
        _shared_clients.pop(project_key, None)
        _close_client(client, "Shared Lean client close failed during restart")

    if len(_shared_clients) >= _MAX_SHARED_CLIENTS:
        _evict_oldest_client()

    client = _start_client(project_key)
    _shared_clients[project_key] = client
    return client


def set_build_in_progress(project_path: Path | str, value: bool) -> None:
    project_key = Path(project_path).resolve(strict=False)
    if value:
        _builds_in_progress.add(project_key)
    else:
        _builds_in_progress.discard(project_key)


def replace_shared_client(
    project_path: Path | str, client: LeanLSPClient | None
) -> LeanLSPClient | None:
    project_key = Path(project_path).resolve(strict=False)
    previous = _shared_clients.pop(project_key, None)
    if client is not None:
        _shared_clients[project_key] = client
    return previous


def close_shared_client(project_path: Path | str | None = None) -> None:
    clients: list[LeanLSPClient] = []

    with CLIENT_LOCK:
        if project_path is None:
            clients = list(_shared_clients.values())
            _shared_clients.clear()
            _builds_in_progress.clear()
        else:
            project_key = Path(project_path).resolve(strict=False)
            client = _shared_clients.pop(project_key, None)
            if client is not None:
                clients.append(client)
            _builds_in_progress.discard(project_key)

    for client in clients:
        _close_client(client, "Shared Lean client close failed during shutdown")


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up."""
    with CLIENT_LOCK:
        configured_root = ctx.request_context.lifespan_context.lean_project_path
        if configured_root is None:
            raise ValueError("lean project path is not set.")
        lean_project_path = bind_lean_project_path(ctx, configured_root)
        client = _get_or_create_shared_client(lean_project_path)
        ctx.request_context.lifespan_context.client = client


def resolve_file_path(
    ctx: Context, file_path: str, *, require_exists: bool = True
) -> Path:
    """Resolve a file path with support for project-root-relative inputs."""
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
    return [
        directory
        for directory in cache_dirs
        if Path(directory).is_relative_to(project_path)
    ]


def infer_project_path(file_path: str, ctx: Context | None = None) -> Path | None:
    """Infer and cache the Lean project path for a file WITHOUT starting the client."""
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

    if ctx and lifespan.lean_project_path:
        try:
            current_policy = build_lean_path_policy(lifespan.lean_project_path)
        except ValueError:
            current_policy = None
        if current_policy is not None and current_policy.contains(resolved_input):
            return cache_project_path(lifespan.lean_project_path, [file_dir])

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
            lifespan.project_cache[current_dir_str] = ""

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
