from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from threading import Lock

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient as BaseLeanLSPClient
from leanclient.file_manager import DiagnosticsResult as BaseDiagnosticsResult

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import OutputCapture


logger = get_logger(__name__)
CLIENT_LOCK = Lock()
SHARED_CLIENT: "LeanLSPClient" | None = None
SHARED_CLIENT_PROJECT_PATH: Path | None = None


class LeanLSPClient(BaseLeanLSPClient):
    """Lean client wrapper with safe cleanup for background diagnostics waits."""

    @staticmethod
    def _with_timed_out_flag(
        result: BaseDiagnosticsResult, timed_out: bool
    ) -> BaseDiagnosticsResult:
        result.timed_out = timed_out
        return result

    @staticmethod
    def _is_benign_detached_future_error(exc: Exception) -> bool:
        msg = str(exc)
        return "The file worker for" in msg and "has been terminated." in msg

    def _detach_async_future(self, future: asyncio.Future) -> None:
        for request_id, candidate in list(self._futures.items()):
            if candidate is future:
                self._futures.pop(request_id, None)
                break

        if not future.done():
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(future.cancel)
            return

        try:
            future.result()
        except asyncio.CancelledError:
            return
        except EOFError as exc:
            logger.debug("Ignoring detached diagnostics wait after LSP shutdown: %s", exc)
        except Exception as exc:
            if self._is_benign_detached_future_error(exc):
                logger.debug(
                    "Ignoring detached diagnostics wait after worker shutdown: %s", exc
                )
            else:
                logger.warning("Detached diagnostics wait failed: %s", exc)

    def _wait_for_diagnostics(
        self,
        uris: list[str],
        inactivity_timeout: float = 15.0,
        max_timeout: float = 300.0,
    ) -> bool:
        """Wait until diagnostics are ready without leaking late waitForDiagnostics futures."""
        paths = [self._uri_to_local(uri) for uri in uris]
        path_by_uri = dict(zip(uris, paths))

        with self._opened_files_lock:
            missing = [p for p in paths if p not in self.opened_files]
            if missing:
                raise FileNotFoundError(
                    f"Files {missing} are not open. Call open_files first."
                )

        uris_needing_wait = []
        target_versions: dict[str, int] = {}
        with self._opened_files_lock:
            for uri in uris:
                path = path_by_uri[uri]
                state = self.opened_files[path]

                if not state.complete and not state.processing:
                    if state.diagnostics_version >= state.version:
                        state.complete = True

                if not state.complete:
                    uris_needing_wait.append(uri)
                    target_versions[uri] = state.version

        if not uris_needing_wait:
            return True

        futures_by_uri: dict[str, asyncio.Future] = {}
        for uri in uris_needing_wait:
            params = {"uri": uri, "version": target_versions[uri]}
            futures_by_uri[uri] = self._send_request_async(
                "textDocument/waitForDiagnostics", params
            )

        def cleanup_wait_futures() -> None:
            for future in futures_by_uri.values():
                self._detach_async_future(future)

        start_time = time.monotonic()
        pending_uris = set(uris_needing_wait)

        try:
            while pending_uris:
                current_time = time.monotonic()
                total_elapsed = current_time - start_time

                completed_uris: set[str] = set()
                max_inactivity = 0.0

                with self._close_condition:
                    for uri in list(pending_uris):
                        future = futures_by_uri[uri]
                        if future.done():
                            path = path_by_uri[uri]
                            state = self.opened_files[path]
                            try:
                                future.result()
                            except Exception as exc:
                                state.error = {"message": str(exc)}
                                state.processing = False
                                state.complete = True
                                state.last_activity = current_time
                                completed_uris.add(uri)
                                continue

                            state.wait_for_diag_done = True
                            if state.is_ready(current_time):
                                state.complete = True
                                completed_uris.add(uri)

                    any_rpc_pending = False
                    for uri in pending_uris - completed_uris:
                        path = path_by_uri[uri]
                        state = self.opened_files[path]

                        if state.is_ready(current_time):
                            state.complete = True
                            completed_uris.add(uri)
                        else:
                            inactivity = current_time - state.last_activity
                            max_inactivity = max(max_inactivity, inactivity)
                            any_rpc_pending = (
                                any_rpc_pending or not futures_by_uri[uri].done()
                            )

                    pending_uris.difference_update(completed_uris)

                    if not pending_uris:
                        return True

                    if self.process.poll() is not None:
                        logger.warning(
                            "_wait_for_diagnostics: LSP process exited (%.1fs total).",
                            total_elapsed,
                        )
                        return False

                    if total_elapsed > max_timeout:
                        logger.warning(
                            "_wait_for_diagnostics hit max timeout of %.1fs (%.1fs total).",
                            max_timeout,
                            total_elapsed,
                        )
                        return False

                    if max_inactivity > inactivity_timeout and not any_rpc_pending:
                        logger.warning(
                            "_wait_for_diagnostics timed out after %.1fs of inactivity (%.1fs total).",
                            inactivity_timeout,
                            total_elapsed,
                        )
                        return False

                    self._close_condition.wait(timeout=0.005)

            return False
        finally:
            cleanup_wait_futures()

    def get_diagnostics(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        inactivity_timeout: float = 15.0,
        max_timeout: float = 300.0,
    ) -> BaseDiagnosticsResult:
        """Return diagnostics with an explicit timed_out flag.

        The upstream leanclient result tracks timeout only indirectly via
        `success=False`. For agentic callers that need to distinguish "file has
        errors" from "Lean stopped responding in time", attach `timed_out`
        directly to the returned DiagnosticsResult instance.
        """
        if start_line is not None and end_line is not None and start_line > end_line:
            raise ValueError("start_line must be <= end_line")

        use_range = start_line is not None or end_line is not None

        with self._opened_files_lock:
            need_to_open = path not in self.opened_files

        if need_to_open:
            self.open_files([path])

        with self._opened_files_lock:
            state = self.opened_files[path]
            if use_range:
                is_complete = (
                    state.is_line_range_complete(start_line, end_line)
                    and state.is_ready()
                )
            else:
                is_complete = state.complete
            uri = state.uri

        wait_completed = True
        if not is_complete:
            if use_range:
                wait_completed = self._wait_for_line_range(
                    [uri], start_line, end_line, inactivity_timeout
                )
            else:
                wait_completed = self._wait_for_diagnostics(
                    [uri],
                    inactivity_timeout=inactivity_timeout,
                    max_timeout=max_timeout,
                )

        with self._opened_files_lock:
            state = self.opened_files[path]

            if state.error:
                return self._with_timed_out_flag(
                    BaseDiagnosticsResult(
                        success=False,
                        diagnostics=[state.error],
                    ),
                    timed_out=not wait_completed,
                )

            has_errors = any(d.get("severity") == 1 for d in state.diagnostics)
            success = not has_errors and not state.fatal_error and wait_completed

            if state.diagnostics:
                filtered = (
                    state.filter_diagnostics_by_range(start_line, end_line)
                    if use_range
                    else state.diagnostics
                )
                return self._with_timed_out_flag(
                    BaseDiagnosticsResult(
                        success=success,
                        diagnostics=filtered,
                    ),
                    timed_out=not wait_completed,
                )

            if state.fatal_error:
                return self._with_timed_out_flag(
                    BaseDiagnosticsResult(
                        success=False,
                        diagnostics=[
                            {
                                "message": "leanclient: Received LeanFileProgressKind.fatalError."
                            }
                        ],
                    ),
                    timed_out=not wait_completed,
                )

            return self._with_timed_out_flag(
                BaseDiagnosticsResult(
                    success=wait_completed,
                    diagnostics=[],
                ),
                timed_out=not wait_completed,
            )


def _close_client_quietly(client: LeanLSPClient | None) -> None:
    if client is None:
        return
    try:
        client.close()
    except Exception:
        logger.exception("Lean client close failed")


def close_shared_client() -> None:
    """Close the process-wide shared Lean client, if any."""
    global SHARED_CLIENT, SHARED_CLIENT_PROJECT_PATH
    with CLIENT_LOCK:
        _close_client_quietly(SHARED_CLIENT)
        SHARED_CLIENT = None
        SHARED_CLIENT_PROJECT_PATH = None


def replace_shared_client(project_path: Path, client: LeanLSPClient) -> None:
    """Replace the process-wide shared Lean client."""
    global SHARED_CLIENT, SHARED_CLIENT_PROJECT_PATH
    with CLIENT_LOCK:
        if SHARED_CLIENT is not None and SHARED_CLIENT is not client:
            _close_client_quietly(SHARED_CLIENT)
        SHARED_CLIENT = client
        SHARED_CLIENT_PROJECT_PATH = project_path.resolve()


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
    """
    global SHARED_CLIENT, SHARED_CLIENT_PROJECT_PATH
    with CLIENT_LOCK:
        lean_project_path = ctx.request_context.lifespan_context.lean_project_path
        if lean_project_path is None:
            raise ValueError("lean project path is not set.")
        lean_project_path = lean_project_path.resolve()

        session_client: LeanLSPClient | None = ctx.request_context.lifespan_context.client

        # Reattach any already-running shared client for this project.
        if SHARED_CLIENT is not None and SHARED_CLIENT_PROJECT_PATH == lean_project_path:
            ctx.request_context.lifespan_context.client = SHARED_CLIENT
            return

        # Drop any stale session-local client handle that is not the shared one.
        if session_client is not None and session_client is not SHARED_CLIENT:
            _close_client_quietly(session_client)

        # Different project path - replace the shared client entirely.
        if SHARED_CLIENT is not None and SHARED_CLIENT_PROJECT_PATH != lean_project_path:
            _close_client_quietly(SHARED_CLIENT)
            SHARED_CLIENT = None
            SHARED_CLIENT_PROJECT_PATH = None

        # Need to create a new client
        # In test environments, prevent repeated cache downloads
        prevent_cache = bool(os.environ.get("LEAN_LSP_TEST_MODE"))
        try:
            with OutputCapture() as output:
                max_files = int(os.environ.get("LEAN_LSP_MAX_OPEN_FILES", "4"))
                client = LeanLSPClient(
                    lean_project_path,
                    initial_build=False,
                    prevent_cache_get=prevent_cache,
                    max_opened_files=max_files,
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

        SHARED_CLIENT = client
        SHARED_CLIENT_PROJECT_PATH = lean_project_path
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
