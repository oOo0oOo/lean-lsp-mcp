import asyncio
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Iterator

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
from lean_lsp_mcp import config


logger = get_logger(__name__)
CLIENT_LOCK = RLock()
_RUNTIME_AVAILABLE = Condition(CLIENT_LOCK)
_project_runtimes: dict[Path, "ProjectRuntime"] = {}


_MAX_SHARED_CLIENTS = 8


class InvalidLeanFilePathError(ValueError):
    pass


@dataclass(frozen=True)
class LspClientOperation:
    client: LeanLSPClient
    project_path: Path


@dataclass(frozen=True)
class LspFileOperation(LspClientOperation):
    path_policy: LeanPathPolicy
    rel_path: str


def _project_key(project_path: Path | str) -> Path:
    return Path(project_path).resolve(strict=False)


def _routing_lock(ctx: Context):
    lifespan = ctx.request_context.lifespan_context
    lock = getattr(lifespan, "routing_lock", None)
    if lock is None:
        lock = RLock()
        lifespan.routing_lock = lock
    return lock


def _set_lifespan_client_if_current(ctx: Context, op: LspClientOperation) -> None:
    set_lifespan_client_for_project(ctx, op.project_path, op.client)


def set_lifespan_client_for_project(
    ctx: Context, project_path: Path | str, client: LeanLSPClient | None
) -> None:
    with _routing_lock(ctx):
        lifespan = ctx.request_context.lifespan_context
        current_root: Path | None = getattr(lifespan, "lean_project_path", None)
        if current_root is not None:
            current_root = current_root.resolve(strict=False)
        if current_root == _project_key(project_path):
            lifespan.client = client


class ProjectRuntime:
    def __init__(self, project_path: Path | str) -> None:
        self.project_path = _project_key(project_path)
        self._client: LeanLSPClient | None = None
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._operation_active = False
        self._build_in_progress = False
        self._registry_reservations = 0

    def _start_or_reuse_client(self) -> LeanLSPClient:
        if self._build_in_progress:
            raise ValueError(
                "A project build is in progress. Retry after the build completes."
            )

        client = self._client
        if client is not None and _client_is_alive(client):
            return client

        if client is not None:
            self._client = None
            _close_client(client, "Shared Lean client close failed during restart")

        self._client = _start_client(self.project_path)
        return self._client

    def _acquire_operation(self) -> LeanLSPClient:
        with self._condition:
            while self._operation_active:
                self._condition.wait()
            if self._build_in_progress:
                raise ValueError(
                    "A project build is in progress. Retry after the build completes."
                )
            client = self._start_or_reuse_client()
            self._operation_active = True
            return client

    def _release_operation(self) -> None:
        with self._condition:
            self._operation_active = False
            self._condition.notify_all()
        with _RUNTIME_AVAILABLE:
            _RUNTIME_AVAILABLE.notify_all()

    def _reserve_for_registry(self) -> None:
        with self._condition:
            self._registry_reservations += 1

    def _release_registry_reservation(self) -> None:
        with self._condition:
            self._registry_reservations -= 1
            self._condition.notify_all()
        with _RUNTIME_AVAILABLE:
            _RUNTIME_AVAILABLE.notify_all()

    @contextmanager
    def operation(self) -> Iterator[LspClientOperation]:
        client = self._acquire_operation()
        try:
            yield LspClientOperation(client=client, project_path=self.project_path)
        finally:
            self._release_operation()

    @contextmanager
    def file_operation(
        self, path_policy: LeanPathPolicy, rel_path: str
    ) -> Iterator[LspFileOperation]:
        client = self._acquire_operation()
        try:
            client.open_file(rel_path)
            yield LspFileOperation(
                client=client,
                project_path=self.project_path,
                path_policy=path_policy,
                rel_path=rel_path,
            )
        finally:
            self._release_operation()

    def detach_for_build(self) -> None:
        with self._condition:
            if self._build_in_progress:
                raise ValueError(
                    "A project build is already in progress. Retry after the build completes."
                )
            while self._operation_active:
                self._condition.wait()
            self._build_in_progress = True
            client = self._client
            self._client = None
        if client is not None:
            _close_client(client, "Lean client close failed during lsp_build restart")

    def install_restarted_client(self, client: LeanLSPClient) -> None:
        with self._condition:
            old_client = self._client
            self._client = client
            self._build_in_progress = False
            self._condition.notify_all()
        if old_client is not None and old_client is not client:
            _close_client(old_client, "Replaced Lean client close failed")

    def clear_build_in_progress(self) -> None:
        with self._condition:
            self._build_in_progress = False
            self._condition.notify_all()
        with _RUNTIME_AVAILABLE:
            _RUNTIME_AVAILABLE.notify_all()

    def can_evict(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            return (
                not self._build_in_progress
                and not self._operation_active
                and self._registry_reservations == 0
            )
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._condition:
            client = self._client
            self._client = None
            self._build_in_progress = False
            self._condition.notify_all()
        if client is not None:
            _close_client(client, "Shared Lean client close failed during shutdown")
        with _RUNTIME_AVAILABLE:
            _RUNTIME_AVAILABLE.notify_all()


def _select_project_runtime(
    project_path: Path | str, *, reserve: bool = False
) -> ProjectRuntime:
    project_key = _project_key(project_path)
    evicted_runtime: ProjectRuntime | None = None
    with CLIENT_LOCK:
        runtime = _project_runtimes.pop(project_key, None)
        if runtime is not None:
            _project_runtimes[project_key] = runtime
            if reserve:
                runtime._reserve_for_registry()
            return runtime

        while len(_project_runtimes) >= _MAX_SHARED_CLIENTS:
            for candidate_key, candidate in list(_project_runtimes.items()):
                if candidate.can_evict():
                    evicted_runtime = _project_runtimes.pop(candidate_key)
                    break
            if evicted_runtime is not None:
                break
            _RUNTIME_AVAILABLE.wait()

        runtime = ProjectRuntime(project_key)
        _project_runtimes[project_key] = runtime
        if reserve:
            runtime._reserve_for_registry()

    if evicted_runtime is not None:
        evicted_runtime.close()

    return runtime


def get_project_runtime(project_path: Path | str) -> ProjectRuntime:
    return _select_project_runtime(project_path)


@contextmanager
def _reserved_project_runtime(project_path: Path | str) -> Iterator[ProjectRuntime]:
    runtime = _select_project_runtime(project_path, reserve=True)
    try:
        yield runtime
    finally:
        runtime._release_registry_reservation()


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


def _is_file_worker_shutdown(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "LSP Error:" in message
        and "-32801" in message
        and "file worker" in message
        and "terminated" in message
    )


def _drain_wait_for_diagnostics_failure(future) -> None:
    try:
        exc = future.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.debug("Failed to inspect waitForDiagnostics future", exc_info=True)
        return

    if exc is None:
        return

    if _is_file_worker_shutdown(exc):
        logger.debug("Ignored stale waitForDiagnostics response: %s", exc)
    else:
        logger.warning("waitForDiagnostics returned an error: %s", exc)


def _install_wait_for_diagnostics_drain(client: LeanLSPClient) -> None:
    if getattr(client, "_lean_lsp_mcp_wait_diag_drain", False):
        return

    send_request_async = getattr(client, "_send_request_async", None)
    if not callable(send_request_async):
        return

    def _send_request_async(method: str, params: dict):
        future = send_request_async(method, params)
        if method == "textDocument/waitForDiagnostics":
            future.add_done_callback(_drain_wait_for_diagnostics_failure)
        return future

    setattr(client, "_send_request_async", _send_request_async)
    setattr(client, "_lean_lsp_mcp_wait_diag_drain", True)


def bind_lean_project_path(ctx: Context, project_path: Path | str) -> Path:
    with _routing_lock(ctx):
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
    with _routing_lock(ctx):
        lifespan = ctx.request_context.lifespan_context
        root = project_path or getattr(lifespan, "lean_project_path", None)
    if root is None:
        raise ValueError("lean project path is not set.")
    return build_lean_path_policy(root)


def _start_client(project_path: Path) -> LeanLSPClient:
    prevent_cache = config.test_mode()
    try:
        with OutputCapture() as output:
            client = LeanLSPClient(
                project_path,  # ty: ignore[invalid-argument-type]
                initial_build=False,
                prevent_cache_get=prevent_cache,
                max_opened_files=_max_opened_files(),
            )
            _install_wait_for_diagnostics_drain(client)
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


def set_build_in_progress(project_path: Path | str, value: bool) -> None:
    runtime = _select_project_runtime(project_path)
    if value:
        runtime.detach_for_build()
    else:
        runtime.clear_build_in_progress()


def replace_shared_client(
    project_path: Path | str, client: LeanLSPClient | None
) -> LeanLSPClient | None:
    runtime = _select_project_runtime(project_path)
    with runtime._condition:
        previous = runtime._client
        runtime._client = client
        if client is not None:
            runtime._build_in_progress = False
            runtime._condition.notify_all()
        return previous


def close_shared_client(project_path: Path | str | None = None) -> None:
    runtimes: list[ProjectRuntime] = []
    with CLIENT_LOCK:
        if project_path is None:
            runtimes = list(_project_runtimes.values())
            _project_runtimes.clear()
        else:
            project_key = _project_key(project_path)
            runtime = _project_runtimes.pop(project_key, None)
            if runtime is not None:
                runtimes.append(runtime)

    for runtime in runtimes:
        runtime.close()


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up."""
    with _routing_lock(ctx):
        configured_root = ctx.request_context.lifespan_context.lean_project_path
        if configured_root is None:
            raise ValueError("lean project path is not set.")
        lean_project_path = bind_lean_project_path(ctx, configured_root)
    with _reserved_project_runtime(lean_project_path) as runtime:
        with runtime.operation() as op:
            _set_lifespan_client_if_current(ctx, op)


@contextmanager
def lsp_client_for_project(ctx: Context) -> Iterator[LspClientOperation]:
    """Yield the current project's shared LSP client for one serialized operation."""
    with _routing_lock(ctx):
        configured_root = ctx.request_context.lifespan_context.lean_project_path
        if configured_root is None:
            raise ValueError("lean project path is not set.")
        project_path = bind_lean_project_path(ctx, configured_root)
    with _reserved_project_runtime(project_path) as runtime:
        with runtime.operation() as op:
            _set_lifespan_client_if_current(ctx, op)
            yield op


def resolve_file_path(
    ctx: Context, file_path: str, *, require_exists: bool = True
) -> Path:
    """Resolve a file path with support for project-root-relative inputs."""
    with _routing_lock(ctx):
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
    if ctx is not None:
        with _routing_lock(ctx):
            return _infer_project_path_locked(file_path, ctx)
    return _infer_project_path_locked(file_path, None)


def _infer_project_path_locked(
    file_path: str, ctx: Context | None = None
) -> Path | None:
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

    with _reserved_project_runtime(project_path) as runtime:
        with runtime.operation() as op:
            _set_lifespan_client_if_current(ctx, op)
    return policy.client_relative_path(resolved_file)


@contextmanager
def lsp_client_for_file(ctx: Context, file_path: str) -> Iterator[LspFileOperation]:
    """Yield a file's shared LSP client for one serialized operation."""
    try:
        resolved_file = str(resolve_file_path(ctx, file_path))
    except (FileNotFoundError, OSError) as exc:
        raise InvalidLeanFilePathError(file_path) from exc

    project_path = infer_project_path(resolved_file, ctx=ctx)
    if project_path is None:
        raise InvalidLeanFilePathError(file_path)

    try:
        policy = build_lean_path_policy(project_path)
    except ValueError as exc:
        raise InvalidLeanFilePathError(file_path) from exc
    if not policy.contains(resolved_file):
        raise InvalidLeanFilePathError(file_path)

    rel_path = policy.client_relative_path(resolved_file)
    with _reserved_project_runtime(project_path) as runtime:
        with runtime.file_operation(path_policy=policy, rel_path=rel_path) as op:
            _set_lifespan_client_if_current(ctx, op)
            yield op
