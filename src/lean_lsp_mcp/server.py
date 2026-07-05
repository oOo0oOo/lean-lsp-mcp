import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
import functools
import json
import logging.config
import os
import re
import ssl
import time
import urllib.request
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, cast

import certifi
import orjson
from leanclient import DocumentContentChange, LeanLSPClient
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.logging import configure_logging, get_logger

from lean_lsp_mcp.client_utils import (
    CLIENT_LOCK as CLIENT_LOCK,
    InvalidLeanFilePathError,
    _reserved_project_runtime,
    _active_transport,
    _max_opened_files,
    _project_switching_allowed,
    get_path_policy as get_path_policy,
    infer_project_path,
    resolve_file_path,
    lsp_client_for_file as lsp_client_for_file,
    lsp_client_for_project as lsp_client_for_project,
    replace_shared_client as replace_shared_client,
    set_build_in_progress as set_build_in_progress,
    setup_client_for_file,
)
from lean_lsp_mcp.file_utils import (
    build_lean_path_policy,
    get_file_contents,
    require_lean_project_path,
)
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp import config
from lean_lsp_mcp.loogle import LoogleManager
from lean_lsp_mcp.repl import Repl, repl_enabled
from lean_lsp_mcp.models import (
    AttemptResult,
    BuildResult,
    DiagnosticMessage,
    # Wrapper models for list-returning tools
    DiagnosticsResult,
    MultiAttemptResult,
    GoalContextEntry,
    StructuredGoal,
)

# REPL models not imported - low-level REPL tools not exposed to keep API simple.
# The model uses lean_multi_attempt which handles REPL internally.
from lean_lsp_mcp.search_utils import check_ripgrep_status, lean_local_search
from lean_lsp_mcp.utils import (
    LeanToolError,
    OutputCapture,
    PreSharedTokenVerifier,
    check_lsp_response,
    extract_failed_dependency_paths,
    extract_goals_list,
    is_build_stderr,
)

# LSP Diagnostic severity: 1=error, 2=warning, 3=info, 4=hint
DIAGNOSTIC_SEVERITY: Dict[int, str] = {1: "error", 2: "warning", 3: "info", 4: "hint"}
_DISABLED_TOOLS_ENV = "LEAN_MCP_DISABLED_TOOLS"
_INSTRUCTIONS_ENV = "LEAN_MCP_INSTRUCTIONS"
_TOOL_DESCRIPTIONS_ENV = "LEAN_MCP_TOOL_DESCRIPTIONS"


def _raise_invalid_path(file_path: str) -> NoReturn:
    """Raise a descriptive error when a file can't be resolved to a Lean project."""
    raise LeanToolError(
        f"Invalid Lean file path: '{file_path}' not found in any Lean project "
        "(no lean-toolchain ancestor or file does not exist)"
    )


def _validate_theorem_name(theorem_name: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*",
        theorem_name,
    ):
        raise LeanToolError(
            "Invalid theorem name. Use a Lean fully qualified name such as `Namespace.theorem`."
        )
    return theorem_name


async def _urlopen_json(req: urllib.request.Request, timeout: float):
    """Run urllib.request.urlopen in a worker thread to avoid blocking the event loop."""
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def _do_request():
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response:
            return orjson.loads(response.read())

    return await asyncio.to_thread(_do_request)


async def _safe_report_progress(
    ctx: Context, *, progress: int, total: int, message: str
) -> None:
    try:
        await ctx.report_progress(progress=progress, total=total, message=message)
    except Exception:
        return


def _parse_disabled_tools(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {name.strip() for name in raw_value.split(",") if name.strip()}


def _load_tool_description_overrides() -> dict[str, str]:
    overrides: dict[str, str] = {}

    inline = config.tool_descriptions_raw()
    if inline:
        try:
            payload = json.loads(inline)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid %s JSON: %s", _TOOL_DESCRIPTIONS_ENV, exc)
        else:
            if not isinstance(payload, dict):
                logger.warning("%s must be a JSON object.", _TOOL_DESCRIPTIONS_ENV)
            else:
                for key, value in payload.items():
                    if isinstance(key, str) and isinstance(value, str):
                        overrides[key] = value

    return overrides


def apply_tool_configuration(server: FastMCP) -> None:
    """Apply optional runtime tool configuration from environment variables."""
    disabled = _parse_disabled_tools(config.disabled_tools_raw())
    for name in sorted(disabled):
        tool = server._tool_manager.get_tool(name)
        if tool is None:
            logger.warning("Cannot disable unknown tool '%s'", name)
            continue
        server.remove_tool(name)
        logger.info("Disabled tool '%s' via %s", name, _DISABLED_TOOLS_ENV)

    instructions_override = config.instructions_override()
    if instructions_override is not None:
        server._mcp_server.instructions = instructions_override
        logger.info("Overrode server instructions via %s", _INSTRUCTIONS_ENV)

    description_overrides = _load_tool_description_overrides()
    for name, description in description_overrides.items():
        tool = server._tool_manager.get_tool(name)
        if tool is None:
            logger.warning("Cannot override description for unknown tool '%s'", name)
            continue
        tool.description = description
        logger.info("Overrode description for '%s'", name)


def _get_build_concurrency_mode() -> str:
    return config.build_concurrency()


_LOG_FILE_CONFIG = config.log_file_config()
_LOG_LEVEL = config.log_level()
if _LOG_FILE_CONFIG:
    try:
        if _LOG_FILE_CONFIG.endswith((".yaml", ".yml")):
            import yaml  # ty: ignore[unresolved-import]

            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            logging.config.dictConfig(cfg)
        elif _LOG_FILE_CONFIG.endswith(".json"):
            with open(_LOG_FILE_CONFIG, "r", encoding="utf-8") as f:
                cfg = orjson.loads(f.read())
            logging.config.dictConfig(cfg)
        else:
            # .ini / fileConfig
            logging.config.fileConfig(_LOG_FILE_CONFIG, disable_existing_loggers=False)
    except Exception as e:
        # fallback to LEAN_LOG_LEVEL so server still runs
        # use the existing configure_logging helper to set level
        configure_logging(cast(Any, "CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL))
        logger = get_logger(__name__)  # temporary to emit the warning
        logger.warning(
            "Failed to load logging config %s: %s. Falling back to LEAN_LOG_LEVEL.",
            _LOG_FILE_CONFIG,
            e,
        )
else:
    configure_logging(cast(Any, "CRITICAL" if _LOG_LEVEL == "NONE" else _LOG_LEVEL))

logger = get_logger(__name__)


_RG_AVAILABLE, _RG_MESSAGE = check_ripgrep_status()


class BuildCoordinator:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._lock = asyncio.Lock()
        self._current_task: asyncio.Task[BuildResult] | None = None

    async def run(
        self, build_factory: Callable[[], Coroutine[Any, Any, BuildResult]]
    ) -> BuildResult:
        if self.mode == "allow":
            return await build_factory()

        async with self._lock:
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            self._current_task = asyncio.create_task(build_factory())
            task = self._current_task

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            # Task was superseded by a newer build
            if self.mode == "cancel":
                return BuildResult(
                    success=False,
                    output="",
                    errors=["Build superseded by newer request."],
                )
            # share: wait for the latest build (follow the chain if it also got superseded)
            while True:
                latest = self._current_task
                try:
                    return await latest
                except asyncio.CancelledError:
                    if self._current_task is latest:
                        raise
                    continue


# ---------------------------------------------------------------------------
# Shared singletons for resources that should NOT be duplicated per-session.
#
# With the ``streamable-http`` transport every MCP session gets its own
# ``app_lifespan`` invocation.  Heavy resources like the local loogle
# subprocess (~6 GB RSS for the Mathlib index) must be initialised exactly
# once and shared across sessions; otherwise N concurrent clients would
# spawn N loogle processes and exhaust memory.
# ---------------------------------------------------------------------------
_shared_loogle_manager: LoogleManager | None = None
_shared_loogle_available: bool = False
_shared_loogle_init_done: bool = False
_shared_loogle_lock = asyncio.Lock()


async def _ensure_shared_loogle(
    lean_project_path: Path | None,
) -> tuple[LoogleManager | None, bool]:
    """Lazily initialise the shared loogle singleton (once, thread-safe)."""
    global _shared_loogle_manager, _shared_loogle_available, _shared_loogle_init_done

    async with _shared_loogle_lock:
        if _shared_loogle_init_done:
            return _shared_loogle_manager, _shared_loogle_available

        if not config.loogle_local_enabled():
            _shared_loogle_init_done = True
            return None, False

        try:
            logger.info("Local loogle enabled, initializing (shared)...")
            manager = _shared_loogle_manager
            if manager is None:
                manager = LoogleManager(project_path=lean_project_path)
                _shared_loogle_manager = manager

            _shared_loogle_available = (
                manager.ensure_installed() and await manager.start()
            )
            if _shared_loogle_available:
                _shared_loogle_init_done = True
                logger.info("Shared local loogle started successfully")
            else:
                logger.warning("Local loogle unavailable, will use remote API")
        except Exception:
            _shared_loogle_available = False
            logger.exception("Local loogle initialization failed, will retry later")
        return _shared_loogle_manager, _shared_loogle_available


@dataclass
class AppContext:
    lean_project_path: Path | None
    client: LeanLSPClient | None
    rate_limit: Dict[str, List[int]]
    lean_search_available: bool
    active_transport: str = "stdio"
    project_switching_allowed: bool = True
    loogle_manager: LoogleManager | None = None
    loogle_local_available: bool = False
    # REPL for efficient multi-attempt execution
    repl: Repl | None = None
    repl_enabled: bool = False
    build_coordinator: BuildCoordinator | None = None


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    repl: Repl | None = None
    context: AppContext | None = None

    try:
        active_transport = _active_transport()
        project_switching_allowed = _project_switching_allowed()
        lean_project_path_str = config.project_path()
        if not lean_project_path_str:
            if not project_switching_allowed:
                raise ValueError(
                    f"`LEAN_PROJECT_PATH` is required when using `{active_transport}` transport."
                )
            lean_project_path = None
        else:
            lean_project_path = require_lean_project_path(lean_project_path_str)

        # Use the shared loogle singleton (initialised at most once)
        loogle_manager, loogle_local_available = await _ensure_shared_loogle(
            lean_project_path
        )

        # Initialize REPL if enabled
        repl_available = False
        if repl_enabled():
            if lean_project_path:
                from lean_lsp_mcp.repl import find_repl_binary

                repl_bin = find_repl_binary(str(lean_project_path))
                if repl_bin:
                    logger.info("REPL enabled, using: %s", repl_bin)
                    repl = Repl(project_dir=str(lean_project_path), repl_path=repl_bin)
                    repl_available = True
                    logger.info("REPL initialized: timeout=%ds", repl.timeout)
                else:
                    logger.warning(
                        "REPL enabled but binary not found. "
                        'Add `require repl from git "https://github.com/leanprover-community/repl"` '
                        "to lakefile and run `lake build repl`. Falling back to LSP."
                    )
            else:
                logger.warning("REPL requires LEAN_PROJECT_PATH to be set")

        build_mode = _get_build_concurrency_mode()
        build_coordinator = BuildCoordinator(build_mode)

        context = AppContext(
            lean_project_path=lean_project_path,
            client=None,
            rate_limit={
                "leansearch": [],
                "loogle": [],
                "leanfinder": [],
                "lean_state_search": [],
                "hammer_premise": [],
            },
            lean_search_available=_RG_AVAILABLE,
            active_transport=active_transport,
            project_switching_allowed=project_switching_allowed,
            loogle_manager=loogle_manager,
            loogle_local_available=loogle_local_available,
            repl=repl,
            repl_enabled=repl_available,
            build_coordinator=build_coordinator,
        )
        yield context
    finally:
        logger.info("Session ending — cleaning up per-session resources")

        # NOTE: Do NOT close context.client here.  The LSP client is a shared
        # singleton managed by client_utils.  Closing it would kill ``lake
        # serve`` for all other sessions.  The shared client is cleaned up via
        # close_shared_client() at process exit (see __init__.py).

        repl_to_close = context.repl if context and context.repl is not None else repl
        if repl_to_close:
            try:
                await repl_to_close.close()
            except Exception:
                logger.exception("REPL close failed during app_lifespan teardown")


mcp_kwargs: dict[str, Any] = dict(
    name="Lean LSP",
    instructions=INSTRUCTIONS,
    dependencies=["leanclient"],
    lifespan=app_lifespan,
)

auth_token = config.auth_token()
if auth_token:
    mcp_kwargs["auth"] = AuthSettings(
        issuer_url=cast(Any, "http://localhost/dummy-issuer"),
        resource_server_url=cast(Any, "http://localhost/dummy-resource"),
    )
    mcp_kwargs["token_verifier"] = PreSharedTokenVerifier(auth_token)

mcp = FastMCP(**mcp_kwargs)

# Symbols imported here but used only by the tool subpackage (via `server.X`)
# or by tests through monkeypatching. Listing them in __all__ marks them as
# intentionally exported so they are not pruned as "unused imports".
__all__ = [
    "setup_client_for_file",
    "resolve_file_path",
    "get_file_contents",
    "lean_local_search",
    "repl_enabled",
    "Repl",
    "LoogleManager",
]


def _custom_backend(env_var: str, default_url: str) -> bool:
    """True when the user configured a self-hosted backend for a tool.

    A custom (non-default) URL means requests do not hit the shared public
    service, so the rate limit no longer applies.
    """
    return config.is_custom_backend(env_var, default_url)


def rate_limited(
    category: str,
    max_requests: int,
    per_seconds: int,
    bypass: Optional[Callable[[], bool]] = None,
):
    def decorator(func):
        def _apply_rate_limit(args, kwargs):
            if bypass is not None and bypass():
                return True, None
            ctx = kwargs.get("ctx")
            if ctx is None:
                if not args:
                    raise KeyError(
                        "rate_limited wrapper requires ctx as a keyword argument or the first positional argument"
                    )
                ctx = args[0]
            rate_limit = ctx.request_context.lifespan_context.rate_limit
            current_time = int(time.time())
            rate_limit[category] = [
                timestamp
                for timestamp in rate_limit[category]
                if timestamp > current_time - per_seconds
            ]
            if len(rate_limit[category]) >= max_requests:
                return (
                    False,
                    f"Tool limit exceeded: {max_requests} requests per {per_seconds} s. Try again later.",
                )
            rate_limit[category].append(current_time)
            return True, None

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return func(*args, **kwargs)

        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f"Limit: {max_requests}req/{per_seconds}s. {doc}"
        return wrapper

    return decorator


async def _close_repl_for_project_switch(app_ctx: AppContext) -> None:
    repl = app_ctx.repl
    app_ctx.repl_enabled = False
    if repl is None:
        return
    app_ctx.repl = None
    try:
        await repl.close()
    except Exception:
        logger.exception("REPL close failed during project switch")


def _set_lifespan_client_for_project(
    ctx: Context, project_path: Path, client: LeanLSPClient | None
) -> None:
    lifespan = ctx.request_context.lifespan_context
    current_root: Path | None = getattr(lifespan, "lean_project_path", None)
    if current_root is not None:
        current_root = current_root.resolve(strict=False)
    if current_root == project_path.resolve(strict=False):
        lifespan.client = client


async def _run_build(
    ctx: Context,
    lean_project_path_obj: Path,
    clean: bool,
    fetch_cache: bool,
    output_lines: int,
) -> BuildResult:
    log_lines: List[str] = []
    errors: List[str] = []
    active_proc: asyncio.subprocess.Process | None = None
    build_flag_set = False

    async def _run_proc(*args: str, **kwargs) -> asyncio.subprocess.Process:
        nonlocal active_proc
        proc = await asyncio.create_subprocess_exec(*args, **kwargs)
        active_proc = proc
        return proc

    async def _handle_build_output_line(line_str: str) -> None:
        line_str = line_str.rstrip()

        if line_str.startswith("trace:") or "LEAN_PATH=" in line_str:
            return

        log_lines.append(line_str)
        if "error" in line_str.lower():
            errors.append(line_str)

        if m := re.search(
            r"\[(\d+)/(\d+)\]\s*(.+?)(?:\s+\(\d+\.?\d*[ms]+\))?$", line_str
        ):
            await _safe_report_progress(
                ctx,
                progress=int(m.group(1)),
                total=int(m.group(2)),
                message=m.group(3) or "Building",
            )

    async def _consume_build_output(proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        remainder = ""
        while chunk := await proc.stdout.read(64 * 1024):
            parts = (remainder + chunk.decode("utf-8", errors="replace")).split("\n")
            remainder = parts.pop()
            for line_str in parts:
                await _handle_build_output_line(line_str)

        if remainder:
            await _handle_build_output_line(remainder)

    runtime = None
    try:
        with _reserved_project_runtime(lean_project_path_obj) as runtime:
            runtime.detach_for_build()
            build_flag_set = True
        _set_lifespan_client_for_project(ctx, lean_project_path_obj, None)

        if clean:
            await _safe_report_progress(
                ctx, progress=1, total=16, message="Running `lake clean`"
            )
            clean_proc = await _run_proc(
                "lake",
                "clean",
                cwd=lean_project_path_obj,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await _consume_build_output(clean_proc)
            await clean_proc.wait()

        if fetch_cache:
            await _safe_report_progress(
                ctx, progress=2, total=16, message="Running `lake exe cache get`"
            )
            cache_proc = await _run_proc(
                "lake",
                "exe",
                "cache",
                "get",
                cwd=lean_project_path_obj,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await _consume_build_output(cache_proc)
            await cache_proc.wait()

        # Run build with progress reporting
        process = await _run_proc(
            "lake",
            "build",
            cwd=lean_project_path_obj,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        await _consume_build_output(process)
        await process.wait()

        if process.returncode != 0:
            return BuildResult(
                success=False,
                output="\n".join(log_lines[-output_lines:]) if output_lines else "",
                errors=errors
                or [f"Build failed with return code {process.returncode}"],
            )

        # Start LSP client (without initial build since we just did it)
        with OutputCapture():
            client = LeanLSPClient(
                lean_project_path_obj,  # ty: ignore[invalid-argument-type]
                initial_build=False,
                prevent_cache_get=True,
                max_opened_files=_max_opened_files(),
            )

        logger.info("Built project and re-started LSP client")
        if runtime is None:
            raise RuntimeError("Lean project runtime was not reserved.")
        runtime.install_restarted_client(client)
        _set_lifespan_client_for_project(ctx, lean_project_path_obj, client)

        return BuildResult(
            success=True,
            output="\n".join(log_lines[-output_lines:]) if output_lines else "",
            errors=[],
        )

    except asyncio.CancelledError:
        if active_proc and active_proc.returncode is None:
            active_proc.terminate()
            try:
                await asyncio.wait_for(active_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                active_proc.kill()
                await active_proc.wait()
        raise
    except Exception as e:
        return BuildResult(
            success=False,
            output="\n".join(log_lines[-output_lines:]) if output_lines else "",
            errors=[str(e)],
        )
    finally:
        if build_flag_set:
            if runtime is not None:
                runtime.clear_build_in_progress()


def _to_diagnostic_messages(diagnostics: Iterable[Dict]) -> List[DiagnosticMessage]:
    """Convert LSP diagnostics to DiagnosticMessage models."""
    result = []
    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue
        severity_int = diag.get("severity", 1)
        result.append(
            DiagnosticMessage(
                severity=DIAGNOSTIC_SEVERITY.get(
                    severity_int, f"unknown({severity_int})"
                ),
                message=diag.get("message", ""),
                line=r["start"]["line"] + 1,
                column=r["start"]["character"] + 1,
            )
        )
    return result


def _process_diagnostics(
    diagnostics: List[Dict],
    build_success: bool,
    severity: Optional[str] = None,
    timed_out: bool = False,
) -> DiagnosticsResult:
    """Process diagnostics, extracting dependency paths from build stderr.

    Args:
        diagnostics: List of diagnostic dicts from leanclient
        build_success: Whether the build succeeded (from leanclient.DiagnosticsResult.success)
        timed_out: Whether the wait timed out (results may be partial)
    """
    items = []
    failed_deps: List[str] = []

    for diag in diagnostics:
        r = diag.get("fullRange", diag.get("range"))
        if r is None:
            continue

        severity_int = diag.get("severity", 1)
        message = diag.get("message", "")
        line = r["start"]["line"] + 1
        column = r["start"]["character"] + 1

        # Check if this is a build failure at (1,1) - extract dependency paths, skip the item
        if line == 1 and column == 1 and is_build_stderr(message):
            failed_deps = extract_failed_dependency_paths(message)
            continue  # Don't include the build stderr blob as a diagnostic item

        # Normal diagnostic from the queried file
        severity_str = DIAGNOSTIC_SEVERITY.get(severity_int, f"unknown({severity_int})")
        if severity is not None and severity_str != severity:
            continue
        items.append(
            DiagnosticMessage(
                severity=severity_str,
                message=message,
                line=line,
                column=column,
            )
        )

    return DiagnosticsResult(
        success=build_success,
        timed_out=timed_out,
        items=items,
        failed_dependencies=failed_deps,
    )


def _goal_to_structured(goal_str: str) -> StructuredGoal:
    goal_str = (goal_str or "").strip()

    # Case: no goals (proof finished)
    if goal_str == "" or goal_str.lower() == "no goals":
        return StructuredGoal(
            context=[],
            goal=None,
            status="complete",
            pretty=goal_str,
        )

    # Case: no turnstile (fallback)
    if "⊢" not in goal_str:
        return StructuredGoal(
            context=[],
            goal=goal_str,
            status="unknown",
            pretty=goal_str,
        )

    before, after = goal_str.split("⊢", 1)

    context: list[GoalContextEntry] = []
    lines = before.splitlines()

    current_name = None
    current_type_lines = []

    def flush():
        nonlocal current_name, current_type_lines
        if current_name is not None:
            context.append(
                GoalContextEntry(
                    name=current_name,
                    type=" ".join(line.strip() for line in current_type_lines).strip(),
                )
            )
        current_name = None
        current_type_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # New hypothesis line
        if ":" in stripped and not line.startswith(" "):
            flush()
            name, typ = stripped.split(":", 1)
            current_name = name.strip()
            current_type_lines = [typ.strip()]
        else:
            # continuation of previous type
            if current_name is not None:
                current_type_lines.append(stripped)

    flush()

    return StructuredGoal(
        context=context,
        goal=after.strip(),
        status="open",
        pretty=goal_str,
    )


def _get_goal_response(
    client: LeanLSPClient, rel_path: str, line: int, column: int
) -> dict | None:
    """Fetch a goal response after waiting for full-file elaboration on cold files."""

    try:
        return client.get_goal(rel_path, line, column)
    except FuturesTimeoutError:
        pass

    client.get_diagnostics(rel_path, inactivity_timeout=30.0)

    try:
        return client.get_goal(rel_path, line, column)
    except FuturesTimeoutError as exc:
        raise LeanToolError("LSP timeout during get_goal") from exc


def _get_line_context(lines: List[str], line: int) -> str:
    """Return the requested line or raise a user-facing range error."""
    if line < 1 or line > len(lines):
        raise LeanToolError(f"Line {line} out of range (file has {len(lines)} lines)")
    return lines[line - 1]


def _resolve_multi_attempt_column(line_context: str, column: Optional[int]) -> int:
    """Resolve a 0-indexed tactic insertion column for multi-attempt edits."""
    if column is None:
        return next((i for i, c in enumerate(line_context) if not c.isspace()), 0)

    if column > len(line_context) + 1:
        raise LeanToolError(
            f"Column {column} out of range for line of length {len(line_context)}"
        )

    return column - 1


def _diagnostic_identity(diagnostic: Dict) -> tuple:
    """Stable identity for a diagnostic — used to compute new vs. baseline
    diagnostics in multi_attempt, so that errors a snippet introduces at
    lines outside the local edit window aren't silently dropped.

    ``code`` and ``source`` are included to disambiguate cases where the
    LSP emits two diagnostics with the same range/severity/message but
    different metadata (e.g. one from a linter, one from the elaborator).
    """
    diagnostic_range = diagnostic.get("range") or diagnostic.get("fullRange") or {}
    start = diagnostic_range.get("start") or {}
    end = diagnostic_range.get("end") or {}
    return (
        start.get("line"),
        start.get("character"),
        end.get("line"),
        end.get("character"),
        diagnostic.get("severity"),
        diagnostic.get("code"),
        diagnostic.get("source"),
        diagnostic.get("message"),
    )


def _shift_baseline_keys(
    baseline_keys: set, edit_start_line: int, line_delta: int
) -> set:
    """Shift baseline diagnostic-identity entries to compensate for the
    file-line shift introduced by a multi-attempt edit.

    Entries at lines strictly before ``edit_start_line`` are unaffected.
    Entries at or beyond it (which the LSP re-emits at line + line_delta
    after the edit) get their start/end lines shifted accordingly so the
    post-edit identity tuple matches.
    """
    if not line_delta:
        return baseline_keys
    shifted = set()
    for key in baseline_keys:
        start_line, start_char, end_line, end_char, severity, code, source, message = (
            key
        )
        if start_line is None or start_line < edit_start_line:
            shifted.add(key)
            continue
        new_start = start_line + line_delta
        new_end = end_line + line_delta if end_line is not None else end_line
        shifted.add(
            (new_start, start_char, new_end, end_char, severity, code, source, message)
        )
    return shifted


def _filter_diagnostics_by_line_range(
    diagnostics: Iterable[Dict], start_line: int, end_line: int
) -> List[Dict]:
    """Return diagnostics that intersect the requested 0-indexed line range."""
    matches: List[Dict] = []
    for diagnostic in diagnostics:
        diagnostic_range = diagnostic.get("range") or diagnostic.get("fullRange")
        if not diagnostic_range:
            continue

        start = diagnostic_range.get("start", {})
        end = diagnostic_range.get("end", {})
        diagnostic_start = start.get("line")
        diagnostic_end = end.get("line")

        if diagnostic_start is None or diagnostic_end is None:
            continue
        if diagnostic_end < start_line or diagnostic_start > end_line:
            continue

        matches.append(diagnostic)

    return matches


def _prepare_multi_attempt_edit(
    line_context: str, target_column: int, snippet: str, total_lines: int, line: int
) -> tuple[str, DocumentContentChange, int, int, int]:
    """Build the temporary edit and return its goal cursor location.

    The final integer is the post-edit line delta: how many lines the file
    grows by once the change is applied. Non-zero only when the snippet is
    near end-of-file and the replacement range is clamped, in which case
    pre-existing diagnostics at lines >= end_line shift by that delta.
    """
    snippet_str = snippet.rstrip("\n")
    snippet_lines = snippet_str.split("\n") if snippet_str else [""]
    indent = line_context[:target_column]
    payload_lines = [
        snippet_lines[0],
        *[f"{indent}{part}" for part in snippet_lines[1:]],
    ]
    payload = "\n".join(payload_lines) + "\n"

    replaced_line_count = max(len(snippet_lines), 1)
    end_line = min(line - 1 + replaced_line_count, total_lines)
    line_delta = len(payload_lines) - (end_line - (line - 1))
    change = DocumentContentChange(
        payload,
        (line - 1, target_column),
        (end_line, 0),
    )

    goal_line = line - 1 + len(payload_lines) - 1
    if len(payload_lines) == 1:
        goal_column = target_column + len(payload_lines[0])
    else:
        goal_column = len(payload_lines[-1])

    return snippet_str, change, goal_line, goal_column, line_delta


async def _multi_attempt_repl(
    ctx: Context,
    file_path: str,
    line: int,
    column: Optional[int] = None,
    snippets: Optional[List[str]] = None,
) -> MultiAttemptResult | None:
    """Try tactics using REPL (fast path)."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    if snippets is None:
        snippets = []
    # Column-aware attempts need the LSP path because the REPL implementation
    # only reconstructs proof state from the start of the target line.
    # Multiline snippets also need the LSP path so diagnostics/goals can be
    # reported at the real end position of the inserted tactic block.
    if (
        column is not None
        or any("\n" in snippet for snippet in snippets)
        or not app_ctx.repl_enabled
        or app_ctx.repl is None
    ):
        return None

    try:
        resolved_path = resolve_file_path(ctx, file_path)
        project_path = infer_project_path(str(resolved_path), ctx=ctx)
        if project_path is None:
            return None
        policy = build_lean_path_policy(project_path)
        resolved_path = policy.validate_path(resolved_path)
        if Path(app_ctx.repl.project_dir).resolve(strict=False) != project_path:
            await _close_repl_for_project_switch(app_ctx)
            return None
        content = get_file_contents(str(resolved_path))
        if content is None:
            return None
        lines = content.splitlines()
        if line > len(lines):
            return None

        base_code = "\n".join(lines[: line - 1])
        repl_results = await app_ctx.repl.run_snippets(base_code, snippets)

        results = []
        for snippet, pr in zip(snippets, repl_results):
            diagnostics = [
                DiagnosticMessage(
                    severity=m.get("severity", "info"),
                    message=m.get("data", ""),
                    line=m.get("pos", {}).get("line", 0),
                    column=m.get("pos", {}).get("column", 0),
                )
                for m in (pr.messages or [])
            ]
            if pr.error:
                diagnostics.append(
                    DiagnosticMessage(
                        severity="error", message=pr.error, line=0, column=0
                    )
                )
            results.append(
                AttemptResult(
                    snippet=snippet.rstrip("\n"),
                    goals=pr.goals or [],
                    diagnostics=diagnostics,
                )
            )
        return MultiAttemptResult(items=results)
    except Exception as e:
        logger.debug(f"REPL multi_attempt failed: {e}")
        return None


def _multi_attempt_lsp(
    ctx: Context,
    file_path: str,
    line: int,
    column: Optional[int] = None,
    snippets: Optional[List[str]] = None,
) -> MultiAttemptResult:
    """Try tactics using LSP file modifications (fallback)."""
    if snippets is None:
        snippets = []
    try:
        with lsp_client_for_file(ctx, file_path) as lsp:
            client = lsp.client
            rel_path = lsp.rel_path
            try:
                original_content = client.get_file_content(rel_path)
            except Exception:
                try:
                    resolved_path = lsp.path_policy.validate_path(
                        resolve_file_path(ctx, file_path)
                    )
                except (FileNotFoundError, ValueError) as exc:
                    raise LeanToolError(str(exc)) from exc
                original_content = get_file_contents(resolved_path)

            lines = (
                original_content.splitlines() if original_content is not None else []
            )
            line_context = _get_line_context(lines, line)
            target_column = _resolve_multi_attempt_column(line_context, column)

            # Snapshot baseline diagnostics before any edit. The line-range filter
            # alone misses errors that surface at distant lines (e.g. a `whnf`
            # heartbeat timeout reported at the leaf-statement line when a snippet's
            # tactic forces aggressive unfolding) — that would produce a misleading
            # `goals=[], diagnostics=[]` result indistinguishable from genuine
            # tactic success. Diff against baseline to surface any *new* diagnostic
            # the snippet introduced, regardless of its line position. Use
            # leanclient's default cutoffs so the baseline wait is symmetric with
            # the per-snippet `get_diagnostics` call below: asymmetric timeouts
            # would leave the baseline partial while the per-snippet snapshot is
            # complete, causing pre-existing diagnostics to be reported as
            # snippet-introduced.
            try:
                baseline_diag = client.get_diagnostics(rel_path)
                check_lsp_response(baseline_diag, "get_diagnostics")
                if getattr(baseline_diag, "timed_out", False):
                    # Baseline is partial — set-diff would over-report. Disable
                    # it and fall back to local line-range filter only.
                    logger.warning(
                        "_multi_attempt_lsp: baseline diagnostics timed out — "
                        "set-diff disabled for this call (results limited to local "
                        "line-range filter)."
                    )
                    baseline_keys = None
                else:
                    baseline_keys = {_diagnostic_identity(d) for d in baseline_diag}
            except Exception:
                logger.warning(
                    "_multi_attempt_lsp: baseline diagnostics unavailable — "
                    "set-diff disabled; results limited to local line-range filter.",
                    exc_info=True,
                )
                baseline_keys = None

            try:
                results: List[AttemptResult] = []
                for i, snippet in enumerate(snippets):
                    # Restore original content before each iteration after the first.
                    # ``_prepare_multi_attempt_edit`` computes edit positions from the
                    # ORIGINAL ``line_context`` / ``len(lines)``; if the previous
                    # snippet's edit grew or shrank the file (EOF clamping), those
                    # positions no longer point to the original sorry region — the
                    # next snippet would patch the wrong content and report drifted
                    # diagnostics as snippet-introduced via ``extra_diag``.
                    if i > 0 and original_content is not None:
                        client.update_file_content(rel_path, original_content)
                    (
                        snippet_str,
                        change,
                        goal_line,
                        goal_column,
                        line_delta,
                    ) = _prepare_multi_attempt_edit(
                        line_context, target_column, snippet, len(lines), line
                    )
                    client.update_file(rel_path, [change])
                    diag = client.get_diagnostics(rel_path)
                    check_lsp_response(diag, "get_diagnostics")
                    filtered_diag = _filter_diagnostics_by_line_range(
                        diag, line - 1, goal_line
                    )
                    in_filtered = {id(d) for d in filtered_diag}
                    # Surface any new diagnostic — relative to the pre-edit baseline —
                    # even if outside the local line range. When baseline capture
                    # failed (baseline_keys is None), set-diff is disabled and we
                    # rely on the local filter only — same behavior as the original
                    # (pre-fix) code path.
                    if baseline_keys is None:
                        extra_diag = []
                    else:
                        # Multi-line snippets near end-of-file can grow the file
                        # (replacement range gets clamped). Pre-existing diagnostics
                        # at or past end_line get re-emitted at shifted line numbers
                        # post-edit, so their identity (which keys on line) won't
                        # match baseline_keys without compensating for the shift.
                        if line_delta:
                            shifted_keys = _shift_baseline_keys(
                                baseline_keys,
                                edit_start_line=line - 1,
                                line_delta=line_delta,
                            )
                        else:
                            shifted_keys = baseline_keys
                        extra_diag = [
                            d
                            for d in diag
                            if id(d) not in in_filtered
                            and _diagnostic_identity(d) not in shifted_keys
                        ]
                    goal_result = client.get_goal(rel_path, goal_line, goal_column)
                    check_lsp_response(goal_result, "get_goal", allow_none=True)
                    goals = extract_goals_list(goal_result)
                    results.append(
                        AttemptResult(
                            snippet=snippet_str,
                            goals=goals,
                            diagnostics=_to_diagnostic_messages(
                                filtered_diag + extra_diag
                            ),
                            timed_out=getattr(diag, "timed_out", False),
                        )
                    )

                return MultiAttemptResult(items=results)
            finally:
                if original_content is not None:
                    try:
                        client.update_file_content(rel_path, original_content)
                    except Exception as exc:
                        logger.warning(
                            "Failed to restore `%s` after multi_attempt: %s",
                            rel_path,
                            exc,
                        )
                    try:
                        # Force a disk resync so transient snippet edits do not leak
                        # into subsequent tool calls for already-open documents.
                        client.open_file(rel_path, force_reopen=True)
                    except Exception as exc:
                        logger.warning(
                            "Failed to force-reopen `%s` after multi_attempt: %s",
                            rel_path,
                            exc,
                        )
    except InvalidLeanFilePathError:
        _raise_invalid_path(file_path)


# Register the tool subpackage last: each submodule's @mcp.tool decorators run
# on import and reference core symbols (helpers, mcp) defined above via this
# module, so the imports must happen after the core is fully defined.
#
# `importlib.import_module` (rather than `from ... import`) is used with a prior
# `sys.modules` pop so that reloading `server` (e.g. via importlib.reload in
# tests) re-executes the submodules and re-registers their tools on the freshly
# created `mcp` instance instead of leaving them bound to the previous one.
import importlib as _importlib  # noqa: E402
import sys as _sys  # noqa: E402

TOOL_MODULES = (
    "build",
    "diagnostics",
    "goals",
    "navigation",
    "search",
    "analysis",
    "widgets",
)
_tool_modules = {}
for _name in TOOL_MODULES:
    _qualified = f"lean_lsp_mcp.tools.{_name}"
    _sys.modules.pop(_qualified, None)
    _tool_modules[_name] = _importlib.import_module(_qualified)

lsp_build = _tool_modules["build"].lsp_build
file_outline = _tool_modules["diagnostics"].file_outline
diagnostic_messages = _tool_modules["diagnostics"].diagnostic_messages
code_actions = _tool_modules["diagnostics"].code_actions
goal = _tool_modules["goals"].goal
term_goal = _tool_modules["goals"].term_goal
hover = _tool_modules["navigation"].hover
completions = _tool_modules["navigation"].completions
declaration_file = _tool_modules["navigation"].declaration_file
references = _tool_modules["navigation"].references
local_search = _tool_modules["search"].local_search
leansearch = _tool_modules["search"].leansearch
loogle = _tool_modules["search"].loogle
leanfinder = _tool_modules["search"].leanfinder
state_search = _tool_modules["search"].state_search
hammer_premise = _tool_modules["search"].hammer_premise
LocalSearchError = _tool_modules["search"].LocalSearchError
multi_attempt = _tool_modules["analysis"].multi_attempt
run_code = _tool_modules["analysis"].run_code
verify_theorem = _tool_modules["analysis"].verify_theorem
minimal_hypotheses = _tool_modules["analysis"].minimal_hypotheses
profile_proof = _tool_modules["analysis"].profile_proof
get_widgets = _tool_modules["widgets"].get_widgets
get_widget_source = _tool_modules["widgets"].get_widget_source


if __name__ == "__main__":
    apply_tool_configuration(mcp)
    os.environ.setdefault(config.ACTIVE_TRANSPORT_ENV, "stdio")
    mcp.run()
