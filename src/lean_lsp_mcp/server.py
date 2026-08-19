import asyncio
import importlib
import importlib.metadata
import json
import logging.config
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import orjson
from leanclient.aio import AsyncLeanLSPClient, LeanClientError
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.utilities.logging import configure_logging, get_logger

from lean_lsp_mcp.attempt_utils import (
    build_attempt_text as _build_attempt_text,
    close_repl_for_project_switch as _close_repl_for_project_switch,
    multi_attempt_lsp as _multi_attempt_lsp,
    multi_attempt_repl as _multi_attempt_repl,
    run_code_repl as _run_code_repl,
)
from lean_lsp_mcp.build_utils import BuildCoordinator, run_build as _run_build
from lean_lsp_mcp.client_utils import (
    _active_transport,
    _project_switching_allowed,
    resolve_file_path,
    setup_client_for_file,
)
from lean_lsp_mcp.diagnostic_utils import (
    diagnostic_identity as _diagnostic_identity,
    filter_diagnostics_by_line_range as _filter_diagnostics_by_line_range,
    get_line_context as _get_line_context,
    goal_strings as _goal_strings,
    goal_to_structured as _goal_to_structured,
    process_diagnostics as _process_diagnostics,
    resolve_multi_attempt_column as _resolve_multi_attempt_column,
    shift_baseline_keys as _shift_baseline_keys,
    to_diagnostic_messages as _to_diagnostic_messages,
)
from lean_lsp_mcp.file_utils import get_file_contents, require_lean_project_path
from lean_lsp_mcp.instructions import INSTRUCTIONS
from lean_lsp_mcp import config
from lean_lsp_mcp.loogle import LoogleManager
from lean_lsp_mcp.repl import Repl, repl_enabled

# REPL models not imported - low-level REPL tools not exposed to keep API simple.
# The model uses lean_multi_attempt which handles REPL internally.
from lean_lsp_mcp.search_utils import check_ripgrep_status, lean_local_search
from lean_lsp_mcp.tool_utils import (
    custom_backend as _custom_backend,
    rate_limited,
    safe_report_progress as _safe_report_progress,
    urlopen_json as _urlopen_json,
)
from lean_lsp_mcp.tool_registry import register_module_tools
from lean_lsp_mcp.utils import LeanToolError, PreSharedTokenVerifier


_DISABLED_TOOLS_ENV = "LEAN_MCP_DISABLED_TOOLS"
_INSTRUCTIONS_ENV = "LEAN_MCP_INSTRUCTIONS"
_TOOL_DESCRIPTIONS_ENV = "LEAN_MCP_TOOL_DESCRIPTIONS"


def _validate_theorem_name(theorem_name: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*",
        theorem_name,
    ):
        raise LeanToolError(
            "Invalid theorem name. Use a Lean fully qualified name such as `Namespace.theorem`."
        )
    return theorem_name


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


def apply_tool_configuration(server: MCPServer) -> None:
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
        server._lowlevel_server.instructions = instructions_override
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
            yaml = cast(Any, importlib.import_module("yaml"))

            with open(_LOG_FILE_CONFIG, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            logging.config.dictConfig(cfg)
        elif _LOG_FILE_CONFIG.endswith(".json"):
            with open(_LOG_FILE_CONFIG, encoding="utf-8") as f:
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
            elif manager.project_path != lean_project_path:
                manager.set_project_path(lean_project_path)

            # ensure_installed() can git-clone + `lake build` for many
            # minutes — keep it off the event loop.
            installed = await asyncio.to_thread(manager.ensure_installed)
            _shared_loogle_available = installed and await manager.start()
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
    client: AsyncLeanLSPClient | None
    rate_limit: dict[str, list[int]]
    lean_search_available: bool
    active_transport: str = "stdio"
    project_switching_allowed: bool = True
    loogle_manager: LoogleManager | None = None
    loogle_local_available: bool = False
    # REPL for efficient multi-attempt execution
    repl: Repl | None = None
    repl_enabled: bool = False
    build_coordinator: BuildCoordinator | None = None
    project_cache: dict[str, Path | str] = field(default_factory=dict)


ToolContext = Context[AppContext, Any]


_prewarm_started = False


async def _prewarm_project_files(project_path: Path, files: list[str]) -> None:
    """Open/elaborate configured files in the background at server startup.

    Overlaps the expensive first elaboration with the agent's initial
    reading/planning phase instead of landing on its first tool call
    (LEAN_MCP_PREWARM_FILES, project-relative, comma-separated).
    """
    from lean_lsp_mcp.client_utils import CLIENT_LOCK, _get_or_create_shared_client

    try:
        async with CLIENT_LOCK:
            client = await _get_or_create_shared_client(project_path)
        opened = []
        for rel_path in files:
            try:
                await client.open(rel_path, wait=False)
                opened.append(rel_path)
            except (OSError, LeanClientError) as exc:
                logger.warning("Prewarm skipped %s: %s", rel_path, exc)
        logger.info("Prewarm started for %s file(s): %s", len(opened), opened)
        for rel_path in opened:
            try:
                await client.barrier(rel_path)
                logger.info("Prewarm complete: %s", rel_path)
            except LeanClientError as exc:
                logger.warning("Prewarm elaboration failed for %s: %s", rel_path, exc)
    except Exception:
        logger.exception("Prewarm task failed")


def _maybe_start_prewarm(lean_project_path: Path | None) -> None:
    global _prewarm_started
    if _prewarm_started or lean_project_path is None:
        return
    files = config.prewarm_files()
    if not files:
        return
    _prewarm_started = True
    asyncio.get_running_loop().create_task(
        _prewarm_project_files(lean_project_path, files)
    )


@asynccontextmanager
async def app_lifespan(server: MCPServer) -> AsyncIterator[AppContext]:
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
        _maybe_start_prewarm(lean_project_path)
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


VERSION = importlib.metadata.version("lean-lsp-mcp")

mcp_kwargs: dict[str, Any] = dict(
    name="Lean LSP",
    instructions=INSTRUCTIONS,
    version=VERSION,
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

mcp = MCPServer(**mcp_kwargs)

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
    "_custom_backend",
    "_build_attempt_text",
    "_close_repl_for_project_switch",
    "_diagnostic_identity",
    "_filter_diagnostics_by_line_range",
    "_get_line_context",
    "_goal_strings",
    "_goal_to_structured",
    "_multi_attempt_lsp",
    "_multi_attempt_repl",
    "_process_diagnostics",
    "_resolve_multi_attempt_column",
    "_run_code_repl",
    "_run_build",
    "_safe_report_progress",
    "_shift_baseline_keys",
    "_to_diagnostic_messages",
    "_urlopen_json",
    "rate_limited",
]


# Import tool modules only after the runtime they depend on is fully defined.
# Their decorators record metadata without binding to a particular MCPServer,
# so reloads can register the same functions on a fresh server explicitly.
from lean_lsp_mcp.tools import analysis as _analysis_tools  # noqa: E402
from lean_lsp_mcp.tools import build as _build_tools  # noqa: E402
from lean_lsp_mcp.tools import diagnostics as _diagnostic_tools  # noqa: E402
from lean_lsp_mcp.tools import goals as _goal_tools  # noqa: E402
from lean_lsp_mcp.tools import navigation as _navigation_tools  # noqa: E402
from lean_lsp_mcp.tools import search as _search_tools  # noqa: E402
from lean_lsp_mcp.tools import widgets as _widget_tools  # noqa: E402

for _tool_module in (
    _build_tools,
    _diagnostic_tools,
    _goal_tools,
    _navigation_tools,
    _search_tools,
    _analysis_tools,
    _widget_tools,
):
    register_module_tools(mcp, _tool_module)

lsp_build = _build_tools.lsp_build
file_outline = _diagnostic_tools.file_outline
diagnostic_messages = _diagnostic_tools.diagnostic_messages
code_actions = _diagnostic_tools.code_actions
goal = _goal_tools.goal
term_goal = _goal_tools.term_goal
hover = _navigation_tools.hover
completions = _navigation_tools.completions
declaration_file = _navigation_tools.declaration_file
references = _navigation_tools.references
local_search = _search_tools.local_search
leansearch = _search_tools.leansearch
loogle = _search_tools.loogle
leanfinder = _search_tools.leanfinder
state_search = _search_tools.state_search
hammer_premise = _search_tools.hammer_premise
LocalSearchError = _search_tools.LocalSearchError
multi_attempt = _analysis_tools.multi_attempt
run_code = _analysis_tools.run_code
verify_theorem = _analysis_tools.verify_theorem
minimal_hypotheses = _analysis_tools.minimal_hypotheses
profile_proof = _analysis_tools.profile_proof
get_widgets = _widget_tools.get_widgets
get_widget_source = _widget_tools.get_widget_source


if __name__ == "__main__":
    apply_tool_configuration(mcp)
    os.environ.setdefault(config.ACTIVE_TRANSPORT_ENV, "stdio")
    mcp.run()
