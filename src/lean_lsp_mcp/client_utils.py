import os
from hashlib import sha256
from pathlib import Path
from threading import Lock

from leanclient import LeanLSPClient
from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger

from lean_lsp_mcp.coordination import COORDINATION_MODE_BROKER, CoordinationError
from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import LeanToolError, OutputCapture


logger = get_logger(__name__)
CLIENT_LOCK = Lock()


def _project_worker_key(project_path: Path, repl_enabled: bool) -> str:
    toolchain_path = project_path / "lean-toolchain"
    toolchain_text = ""
    if toolchain_path.exists():
        try:
            toolchain_text = toolchain_path.read_text(encoding="utf-8").strip()
        except Exception:
            toolchain_text = "<unreadable>"
    toolchain_hash = sha256(toolchain_text.encode("utf-8")).hexdigest()[:16]
    repl_flag = "1" if repl_enabled else "0"
    return f"{project_path.resolve()}::tc={toolchain_hash}::repl={repl_flag}"


def _release_lease(lifespan: object) -> bool:
    lease_id = getattr(lifespan, "client_lease_id", None)
    if not lease_id:
        return True

    coordination_client = getattr(lifespan, "coordination_client", None)
    instance_id = str(getattr(lifespan, "instance_id", ""))
    if coordination_client is None or not instance_id:
        logger.error(
            "Cannot release coordination lease: missing coordination client or instance id."
        )
        return False

    try:
        coordination_client.release_lease(instance_id=instance_id, lease_id=lease_id)
    except Exception:
        logger.exception("Failed to release coordination lease")
        return False

    setattr(lifespan, "client_lease_id", None)
    setattr(lifespan, "client_worker_key", None)
    return True


def _release_lease_by_id(
    *,
    coordination_client: object,
    instance_id: str,
    lease_id: str,
    reason: str,
) -> bool:
    try:
        coordination_client.release_lease(instance_id=instance_id, lease_id=lease_id)
    except Exception:
        logger.exception("Failed to release coordination lease (%s)", reason)
        return False
    return True


def _create_client_locked(
    lifespan: object,
    lean_project_path: Path,
    *,
    prevent_cache_get_override: bool | None,
) -> None:
    # In test environments, prevent repeated cache downloads.
    prevent_cache = bool(os.environ.get("LEAN_LSP_TEST_MODE"))
    if prevent_cache_get_override is not None:
        prevent_cache = prevent_cache_get_override

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
        setattr(lifespan, "client", client)
    except Exception:
        # Fail closed: in broker mode multiple startup calls can race with
        # idempotent acquire_lease. Releasing here can drop a lease that
        # another in-flight call is about to use.
        raise


def _close_client_locked(
    lifespan: object,
    *,
    release_lease: bool = True,
    require_lease_release: bool = False,
) -> bool:
    client = getattr(lifespan, "client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            logger.exception("Lean client close failed")
            return False

    setattr(lifespan, "client", None)
    if not release_lease:
        return True

    lease_released = _release_lease(lifespan)
    if require_lease_release and not lease_released:
        return False
    return True


def close_client(ctx: Context, *, release_lease: bool = True) -> bool:
    """Close the current Lean client.

    Args:
        ctx: MCP tool context.
        release_lease: If True, also release any broker lease held by this context.
    """
    with CLIENT_LOCK:
        return _close_client_locked(
            ctx.request_context.lifespan_context,
            release_lease=release_lease,
            require_lease_release=release_lease,
        )


def startup_client(ctx: Context, prevent_cache_get_override: bool | None = None):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
        prevent_cache_get_override: Optional explicit cache fetch behavior.
    """
    while True:
        lease_request: tuple[object, str, str, str, Path] | None = None

        with CLIENT_LOCK:
            lifespan = ctx.request_context.lifespan_context
            lean_project_path = lifespan.lean_project_path
            if lean_project_path is None:
                raise ValueError("lean project path is not set.")

            # Check if already correct client.
            client: LeanLSPClient | None = lifespan.client
            if client is not None:
                if client.project_path == lean_project_path:
                    return
                if not _close_client_locked(lifespan):
                    raise LeanToolError(
                        "Failed to close existing Lean client; refusing to switch "
                        "projects while shutdown state is uncertain."
                    )

            coordination_mode = str(getattr(lifespan, "coordination_mode", "")).strip()
            if coordination_mode == COORDINATION_MODE_BROKER:
                coordination_client = getattr(lifespan, "coordination_client", None)
                if coordination_client is None:
                    raise LeanToolError(
                        "Broker coordination is enabled but broker client is unavailable."
                    )
                worker_key = _project_worker_key(
                    lean_project_path, bool(getattr(lifespan, "repl_enabled", False))
                )
                instance_id = str(getattr(lifespan, "instance_id", ""))
                lineage_root = str(getattr(lifespan, "lineage_root", ""))
                existing_lease_id = str(getattr(lifespan, "client_lease_id", "") or "")
                existing_worker_key = str(getattr(lifespan, "client_worker_key", "") or "")

                if existing_lease_id and existing_worker_key != worker_key:
                    if not _release_lease(lifespan):
                        raise LeanToolError(
                            "Failed to release existing coordinated Lean worker lease; "
                            "cannot switch projects while lease state is uncertain."
                        )
                    existing_lease_id = ""

                lease_request = (
                    coordination_client,
                    instance_id,
                    lineage_root,
                    worker_key,
                    lean_project_path,
                )

            if lease_request is None:
                _create_client_locked(
                    lifespan,
                    lean_project_path,
                    prevent_cache_get_override=prevent_cache_get_override,
                )
                return

        coordination_client, instance_id, lineage_root, worker_key, requested_project = (
            lease_request
        )
        try:
            lease_id = coordination_client.acquire_lease(
                instance_id=instance_id,
                lineage_root=lineage_root,
                worker_key=worker_key,
            )
        except CoordinationError as exc:
            raise LeanToolError(
                f"Failed to acquire coordinated Lean worker lease: {exc}"
            ) from exc

        with CLIENT_LOCK:
            lifespan = ctx.request_context.lifespan_context
            current_project = getattr(lifespan, "lean_project_path", None)
            current_client: LeanLSPClient | None = getattr(lifespan, "client", None)
            if current_project != requested_project:
                released = _release_lease_by_id(
                    coordination_client=coordination_client,
                    instance_id=instance_id,
                    lease_id=lease_id,
                    reason="stale startup request",
                )
                if not released:
                    raise LeanToolError(
                        "Failed to release stale coordinated Lean worker lease after "
                        "project change; refusing to continue while lease state is uncertain."
                    )
                continue
            if current_client is not None:
                if current_client.project_path == requested_project:
                    setattr(lifespan, "client_lease_id", lease_id)
                    setattr(lifespan, "client_worker_key", worker_key)
                    return
                released = _release_lease_by_id(
                    coordination_client=coordination_client,
                    instance_id=instance_id,
                    lease_id=lease_id,
                    reason="client switched while waiting for lease",
                )
                if not released:
                    raise LeanToolError(
                        "Failed to release stale coordinated Lean worker lease after "
                        "client switch; refusing to continue while lease state is uncertain."
                    )
                continue

            existing_lease_id = str(getattr(lifespan, "client_lease_id", "") or "")
            existing_worker_key = str(getattr(lifespan, "client_worker_key", "") or "")
            if existing_lease_id and existing_worker_key != worker_key:
                released = _release_lease_by_id(
                    coordination_client=coordination_client,
                    instance_id=instance_id,
                    lease_id=lease_id,
                    reason="competing lease assignment",
                )
                if not released:
                    raise LeanToolError(
                        "Failed to release stale coordinated Lean worker lease after "
                        "competing assignment; refusing to continue while lease state is uncertain."
                    )
                continue

            setattr(lifespan, "client_lease_id", lease_id)
            setattr(lifespan, "client_worker_key", worker_key)

            _create_client_locked(
                lifespan,
                requested_project,
                prevent_cache_get_override=prevent_cache_get_override,
            )
            return


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
