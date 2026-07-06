import asyncio
from pathlib import Path

from leanclient.aio import AsyncLeanLSPClient, ScratchPool
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger

from lean_lsp_mcp.file_utils import (
    LeanPathPolicy,
    build_lean_path_policy,
    require_lean_project_path,
    resolve_input_path,
    valid_lean_project_path,
)
from lean_lsp_mcp import config


logger = get_logger(__name__)
CLIENT_LOCK = asyncio.Lock()
_shared_clients: dict[Path, AsyncLeanLSPClient] = {}
_shared_pools: dict[Path, ScratchPool] = {}
_builds_in_progress: set[Path] = set()


_MAX_SHARED_CLIENTS = 8


def _active_transport(ctx: Context | None = None) -> str:
    if ctx is not None:
        lifespan = ctx.request_context.lifespan_context
        transport = getattr(lifespan, "active_transport", None)
        if isinstance(transport, str) and transport:
            return transport
    return config.active_transport()


def _project_switching_allowed(ctx: Context | None = None) -> bool:
    if ctx is not None:
        lifespan = ctx.request_context.lifespan_context
        explicit = getattr(lifespan, "project_switching_allowed", None)
        if explicit is not None:
            return bool(explicit)
    return _active_transport(ctx) == "stdio"


def _max_opened_files() -> int:
    return config.max_open_files()


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
        current_client: AsyncLeanLSPClient | None = getattr(lifespan, "client", None)
        if current_client is not None and Path(
            getattr(current_client, "project_path", "")
        ) != resolved_project:
            lifespan.client = None

    return resolved_project


def get_path_policy(ctx: Context, project_path: Path | None = None) -> LeanPathPolicy:
    lifespan = ctx.request_context.lifespan_context
    root = project_path or getattr(lifespan, "lean_project_path", None)
    if root is None:
        raise ValueError("lean project path is not set.")
    return build_lean_path_policy(root)


async def _start_client(project_path: Path) -> AsyncLeanLSPClient:
    try:
        client = AsyncLeanLSPClient(
            str(project_path),
            max_workers=_max_opened_files(),
        )
        await client.start()
        logger.info("Shared async LSP client connected at %s", project_path)
    except Exception as exc:
        logger.exception("Failed to start shared Lean LSP client")
        raise ValueError(
            f"Failed to start Lean language server at '{project_path}': {exc}"
        ) from exc
    return client


async def _evict_oldest_client() -> None:
    """Close and remove the oldest shared client to stay within the cap."""
    oldest_key = next(iter(_shared_clients))
    old = _shared_clients.pop(oldest_key)
    _shared_pools.pop(oldest_key, None)
    try:
        await old.close()
    except Exception:
        logger.exception("Evicted shared client close failed for %s", oldest_key)


async def _get_or_create_shared_client(
    lean_project_path: Path,
) -> AsyncLeanLSPClient:
    project_key = lean_project_path.resolve(strict=False)

    if project_key in _builds_in_progress:
        raise ValueError(
            "A project build is in progress. Retry after the build completes."
        )

    client = _shared_clients.get(project_key)
    if client is not None and client.alive:
        return client

    if client is not None:
        _shared_clients.pop(project_key, None)
        _shared_pools.pop(project_key, None)
        try:
            await client.close()
        except Exception:
            logger.exception("Shared Lean client close failed during restart")

    if len(_shared_clients) >= _MAX_SHARED_CLIENTS:
        await _evict_oldest_client()

    client = await _start_client(project_key)
    _shared_clients[project_key] = client
    return client


def get_scratch_pool(ctx: Context) -> ScratchPool:
    """Per-project pre-warmed virtual-document pool for snippet trials.

    Slots warm lazily with empty content; the first trial that imports
    Mathlib pays the import once per slot, later trials reuse the header
    snapshot (same-prefix didChange).
    """
    lifespan = ctx.request_context.lifespan_context
    project = lifespan.lean_project_path
    if project is None:
        raise ValueError("lean project path is not set.")
    project_key = Path(project).resolve(strict=False)
    pool = _shared_pools.get(project_key)
    client = _shared_clients.get(project_key)
    if client is None:
        raise ValueError("Lean client is not running for this project.")
    if pool is None or pool._client is not client:
        pool = ScratchPool(
            client,
            header="",
            size=config.scratch_pool_size(),
            name_prefix="_mcp_scratch",
        )
        _shared_pools[project_key] = pool
    return pool


def set_build_in_progress(project_path: Path | str, value: bool) -> None:
    project_key = Path(project_path).resolve(strict=False)
    if value:
        _builds_in_progress.add(project_key)
    else:
        _builds_in_progress.discard(project_key)


async def detach_shared_client(
    project_path: Path | str,
) -> AsyncLeanLSPClient | None:
    """Remove (without closing) the shared client for a project."""
    project_key = Path(project_path).resolve(strict=False)
    _shared_pools.pop(project_key, None)
    return _shared_clients.pop(project_key, None)


def attach_shared_client(project_path: Path | str, client: AsyncLeanLSPClient) -> None:
    project_key = Path(project_path).resolve(strict=False)
    _shared_clients[project_key] = client


def close_shared_client(project_path: Path | str | None = None) -> None:
    """Terminate shared clients synchronously (process-exit path).

    Safe to call after the event loop has closed: kills the ``lake serve``
    process groups directly instead of awaiting a graceful close.
    """
    if project_path is None:
        clients = list(_shared_clients.values())
        _shared_clients.clear()
        _shared_pools.clear()
        _builds_in_progress.clear()
    else:
        project_key = Path(project_path).resolve(strict=False)
        clients = []
        client = _shared_clients.pop(project_key, None)
        if client is not None:
            clients.append(client)
        _shared_pools.pop(project_key, None)
        _builds_in_progress.discard(project_key)

    for client in clients:
        try:
            client._transport._kill_group()
        except Exception:
            logger.exception("Shared Lean client terminate failed during shutdown")


async def startup_client(ctx: Context) -> AsyncLeanLSPClient:
    """Ensure the shared async Lean client for the session's project is up."""
    async with CLIENT_LOCK:
        configured_root = ctx.request_context.lifespan_context.lean_project_path
        if configured_root is None:
            raise ValueError("lean project path is not set.")
        lean_project_path = bind_lean_project_path(ctx, configured_root)
        client = await _get_or_create_shared_client(lean_project_path)
        ctx.request_context.lifespan_context.client = client
        return client


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


async def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
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

    await startup_client(ctx)
    return policy.client_relative_path(resolved_file)


async def open_synced(ctx: Context, rel_path: str, wait: bool = False):
    """Open the file and sync it with the current on-disk content."""
    client: AsyncLeanLSPClient = ctx.request_context.lifespan_context.client
    return await client.reload_from_disk(rel_path, wait=wait)
