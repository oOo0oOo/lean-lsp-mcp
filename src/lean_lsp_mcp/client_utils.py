from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from leanclient.aio import AsyncLeanLSPClient, ScratchPool
from mcp.server.mcpserver.utilities.logging import get_logger

from lean_lsp_mcp.file_utils import (
    LeanPathPolicy,
    build_lean_path_policy,
    require_lean_project_path,
    resolve_input_path,
    valid_lean_project_path,
)
from lean_lsp_mcp import config
from lean_lsp_mcp.utils import LeanToolError

if TYPE_CHECKING:
    from lean_lsp_mcp.server import ToolContext


logger = get_logger(__name__)
CLIENT_LOCK = asyncio.Lock()


@dataclass
class ProjectRuntime:
    """Shared LSP resources owned by one Lean project."""

    client: AsyncLeanLSPClient | None = None
    scratch_pool: ScratchPool | None = None
    serial_pool: ScratchPool | None = None
    build_in_progress: bool = False


_project_runtimes: dict[Path, ProjectRuntime] = {}


_MAX_SHARED_CLIENTS = 8


def _project_key(project_path: Path | str) -> Path:
    return Path(project_path).resolve(strict=False)


def _project_runtime(project_path: Path | str) -> ProjectRuntime:
    return _project_runtimes.setdefault(_project_key(project_path), ProjectRuntime())


def _discard_empty_runtime(project_key: Path) -> None:
    runtime = _project_runtimes.get(project_key)
    if (
        runtime is not None
        and runtime.client is None
        and runtime.scratch_pool is None
        and runtime.serial_pool is None
        and not runtime.build_in_progress
    ):
        _project_runtimes.pop(project_key, None)


def _active_transport(ctx: ToolContext | None = None) -> str:
    if ctx is not None:
        lifespan = ctx.request_context.lifespan_context
        transport = getattr(lifespan, "active_transport", None)
        if isinstance(transport, str) and transport:
            return transport
    return config.active_transport()


def _project_switching_allowed(ctx: ToolContext | None = None) -> bool:
    if ctx is not None:
        lifespan = ctx.request_context.lifespan_context
        explicit = getattr(lifespan, "project_switching_allowed", None)
        if explicit is not None:
            return bool(explicit)
    return _active_transport(ctx) == "stdio"


def _max_opened_files() -> int:
    return config.max_open_files()


def bind_lean_project_path(ctx: ToolContext, project_path: Path | str) -> Path:
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
        if (
            current_client is not None
            and Path(getattr(current_client, "project_path", "")) != resolved_project
        ):
            lifespan.client = None

    return resolved_project


def get_path_policy(
    ctx: ToolContext, project_path: Path | None = None
) -> LeanPathPolicy:
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
    oldest_key = next(
        key for key, runtime in _project_runtimes.items() if runtime.client is not None
    )
    runtime = _project_runtimes.pop(oldest_key)
    old = runtime.client
    assert old is not None
    try:
        await old.close()
    except Exception:
        logger.exception("Evicted shared client close failed for %s", oldest_key)


async def _get_or_create_shared_client(
    lean_project_path: Path,
) -> AsyncLeanLSPClient:
    project_key = _project_key(lean_project_path)
    runtime = _project_runtimes.get(project_key)

    if runtime is not None and runtime.build_in_progress:
        raise ValueError(
            "A project build is in progress. Retry after the build completes."
        )

    client = runtime.client if runtime is not None else None
    if client is not None and client.alive:
        return client

    if client is not None:
        assert runtime is not None
        runtime.client = None
        runtime.scratch_pool = None
        runtime.serial_pool = None
        try:
            await client.close()
        except Exception:
            logger.exception("Shared Lean client close failed during restart")

    client_count = sum(
        runtime.client is not None for runtime in _project_runtimes.values()
    )
    if client_count >= _MAX_SHARED_CLIENTS:
        await _evict_oldest_client()

    client = await _start_client(project_key)
    _project_runtime(project_key).client = client
    return client


def _get_scratch_pool(ctx: ToolContext, *, serial: bool) -> ScratchPool:
    lifespan = ctx.request_context.lifespan_context
    project = lifespan.lean_project_path
    if project is None:
        raise ValueError("lean project path is not set.")
    runtime = _project_runtimes.get(_project_key(project))
    if runtime is None or runtime.client is None:
        raise ValueError("Lean client is not running for this project.")
    client = runtime.client

    pool = runtime.serial_pool if serial else runtime.scratch_pool
    if pool is None or pool._client is not client:
        pool = ScratchPool(
            client,
            header="",
            size=1 if serial else config.scratch_pool_size(),
            name_prefix="_mcp_serial" if serial else "_mcp_scratch",
        )
        if serial:
            runtime.serial_pool = pool
        else:
            runtime.scratch_pool = pool
    return pool


def get_scratch_pool(ctx: ToolContext) -> ScratchPool:
    """Per-project pre-warmed virtual-document pool for parallel trials.

    Slots warm lazily with empty content; the first trial that imports
    Mathlib pays the import once per slot, later trials reuse the header
    snapshot (same-prefix didChange).
    """
    return _get_scratch_pool(ctx, serial=False)


def get_serial_scratch_pool(ctx: ToolContext) -> ScratchPool:
    """Single-slot pool for scratch tools that issue one trial at a time.

    Keeping sequential work separate from the parallel trial pool prevents
    alternating calls from warming two copies of the same import environment.
    """
    return _get_scratch_pool(ctx, serial=True)


def get_run_code_pool(ctx: ToolContext) -> ScratchPool:
    """Backward-compatible name for the sequential scratch pool."""
    return get_serial_scratch_pool(ctx)


def set_build_in_progress(project_path: Path | str, value: bool) -> None:
    project_key = _project_key(project_path)
    if value:
        _project_runtime(project_key).build_in_progress = True
    else:
        runtime = _project_runtimes.get(project_key)
        if runtime is not None:
            runtime.build_in_progress = False
            _discard_empty_runtime(project_key)


async def detach_shared_client(
    project_path: Path | str,
) -> AsyncLeanLSPClient | None:
    """Remove (without closing) the shared client for a project."""
    project_key = _project_key(project_path)
    runtime = _project_runtimes.get(project_key)
    if runtime is None:
        return None
    client = runtime.client
    runtime.client = None
    runtime.scratch_pool = None
    runtime.serial_pool = None
    _discard_empty_runtime(project_key)
    return client


def attach_shared_client(project_path: Path | str, client: AsyncLeanLSPClient) -> None:
    runtime = _project_runtime(project_path)
    if runtime.client is not client:
        runtime.scratch_pool = None
        runtime.serial_pool = None
    runtime.client = client


def running_shared_client(project_path: Path | str) -> AsyncLeanLSPClient | None:
    """Return the shared client for *project_path*, but only if one is already up.

    Unlike :func:`startup_client` this never launches ``lake serve``. Callers
    that only enrich a result with language server data use it so that a fast
    operation cannot silently turn into a cold server start.
    """
    runtime = _project_runtimes.get(_project_key(project_path))
    client = runtime.client if runtime is not None else None
    if client is None or not client.alive:
        return None
    return client


def close_shared_client(project_path: Path | str | None = None) -> None:
    """Terminate shared clients synchronously (process-exit path).

    Safe to call after the event loop has closed: kills the ``lake serve``
    process groups directly instead of awaiting a graceful close.
    """
    if project_path is None:
        clients = [
            runtime.client
            for runtime in _project_runtimes.values()
            if runtime.client is not None
        ]
        _project_runtimes.clear()
    else:
        runtime = _project_runtimes.pop(_project_key(project_path), None)
        clients = [runtime.client] if runtime is not None and runtime.client else []

    for client in clients:
        try:
            client._transport._kill_group()
        except Exception:
            logger.exception("Shared Lean client terminate failed during shutdown")


async def startup_client(ctx: ToolContext) -> AsyncLeanLSPClient:
    """Ensure the shared async Lean client for the session's project is up."""
    async with CLIENT_LOCK:
        configured_root = ctx.request_context.lifespan_context.lean_project_path
        if configured_root is None:
            raise ValueError("lean project path is not set.")
        lean_project_path = bind_lean_project_path(ctx, configured_root)
        client = await _get_or_create_shared_client(lean_project_path)
        ctx.request_context.lifespan_context.client = client
        return client


def get_client(ctx: ToolContext) -> AsyncLeanLSPClient:
    client = ctx.request_context.lifespan_context.client
    if client is None:
        raise ValueError("Lean client is not running for this project.")
    return client


def resolve_file_path(
    ctx: ToolContext, file_path: str, *, require_exists: bool = True
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


def _cache_project_path(
    ctx: ToolContext | None, project_path: Path, cache_dirs: list[str]
) -> Path:
    if ctx is None:
        return project_path

    bound_project = bind_lean_project_path(ctx, project_path)
    cache_targets = _cacheable_project_dirs(bound_project, cache_dirs)
    project_cache = ctx.request_context.lifespan_context.project_cache
    for directory in {*cache_targets, str(bound_project)}:
        if directory:
            project_cache[directory] = bound_project
    return bound_project


def _cached_project_path(
    ctx: ToolContext | None, directory: str, resolved_input: Path
) -> Path | None:
    if ctx is None:
        return None

    project_cache = ctx.request_context.lifespan_context.project_cache
    cached_root = project_cache.get(directory)
    if not cached_root:
        return None
    try:
        policy = build_lean_path_policy(Path(cached_root))
    except ValueError:
        project_cache[directory] = ""
        return None
    if policy.contains(resolved_input):
        return Path(cached_root)
    project_cache[directory] = ""
    return None


def _bound_project_path(
    ctx: ToolContext | None, resolved_input: Path, file_dir: str
) -> Path | None:
    if ctx is None:
        return None
    project_path = ctx.request_context.lifespan_context.lean_project_path
    if not project_path:
        return None
    try:
        policy = build_lean_path_policy(project_path)
    except ValueError:
        return None
    if not policy.contains(resolved_input):
        return None
    return _cache_project_path(ctx, project_path, [file_dir])


def infer_project_path(file_path: str, ctx: ToolContext | None = None) -> Path | None:
    """Infer and cache the Lean project path for a file WITHOUT starting the client."""
    if ctx is not None:
        resolved_input = resolve_file_path(ctx, file_path, require_exists=False)
    else:
        resolved_input = resolve_input_path(file_path, require_exists=False)

    start_dir = resolved_input if resolved_input.is_dir() else resolved_input.parent
    start_dir = start_dir.resolve(strict=False)
    file_dir = str(start_dir)

    lifespan = ctx.request_context.lifespan_context if ctx is not None else None
    if bound_project := _bound_project_path(ctx, resolved_input, file_dir):
        return bound_project

    current_dir = start_dir
    cache_dirs: list[str] = []
    candidates: list[Path] = []
    while True:
        current_dir_str = str(current_dir)
        cache_dirs.append(current_dir_str)

        if cached_root := _cached_project_path(ctx, current_dir_str, resolved_input):
            return _cache_project_path(ctx, cached_root, cache_dirs)

        if valid_lean_project_path(current_dir):
            candidates.append(current_dir.resolve(strict=True))
        elif lifespan is not None:
            lifespan.project_cache[current_dir_str] = ""

        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent

    if chosen_root := _pick_project_root(resolved_input, candidates):
        return _cache_project_path(ctx, chosen_root, cache_dirs)

    return None


async def setup_client_for_file(ctx: ToolContext, file_path: str) -> str | None:
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


async def require_client_for_file(
    ctx: ToolContext, file_path: str, *, error_message: str | None = None
) -> str:
    """Prepare a file's project client or raise the standard tool-path error."""
    rel_path = await setup_client_for_file(ctx, file_path)
    if rel_path is None:
        if error_message is not None:
            raise LeanToolError(error_message)
        raise LeanToolError(
            f"Invalid Lean file path: '{file_path}' not found in any Lean project "
            "(no lean-toolchain ancestor or file does not exist)"
        )
    return rel_path


async def open_synced(ctx: ToolContext, rel_path: str, wait: bool = False):
    """Open the file and sync it with the current on-disk content."""
    client = get_client(ctx)
    return await client.reload_from_disk(rel_path, wait=wait)
