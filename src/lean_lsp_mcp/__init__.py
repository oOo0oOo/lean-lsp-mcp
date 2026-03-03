import argparse
import os
import sys
import uuid
from contextlib import suppress
from pathlib import Path

import anyio
from lean_lsp_mcp.client_utils import infer_project_path
from lean_lsp_mcp.coordination import (
    COORDINATION_MODE_BROKER,
    COORDINATION_MODE_DIRECT,
    DEFAULT_MAX_LINEAGE_DEPTH,
    DEFAULT_MAX_WORKERS,
    ENV_COORDINATION_DIR,
    ENV_COORDINATION_MODE,
    ENV_INSTANCE_ID,
    ENV_LINEAGE_DEPTH,
    ENV_LINEAGE_ROOT,
    ENV_MAX_LINEAGE_DEPTH,
    ENV_MAX_WORKERS,
    CoordinationError,
    default_coordination_dir,
    derive_lineage,
    parse_non_negative_int_env,
)
from lean_lsp_mcp.server import mcp

_TRANSPORT_CLOSE_HINTS = (
    "transport closed",
    "connection closed",
    "broken pipe",
)

_TRANSPORT_CLOSE_EXCEPTIONS = (
    BrokenPipeError,
    ConnectionResetError,
    EOFError,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
)


def _broker_coordination_supported() -> bool:
    return os.name != "nt"


def _resolve_coordination_mode(
    parser: argparse.ArgumentParser, cli_value: str | None
) -> str:
    if cli_value:
        return cli_value

    env_value = os.environ.get(ENV_COORDINATION_MODE, "").strip()
    if not env_value:
        return COORDINATION_MODE_DIRECT
    if env_value not in (COORDINATION_MODE_DIRECT, COORDINATION_MODE_BROKER):
        parser.error(
            f"{ENV_COORDINATION_MODE} must be one of "
            f"{COORDINATION_MODE_DIRECT!r} or {COORDINATION_MODE_BROKER!r}, got: {env_value!r}"
        )
    return env_value


def _resolve_non_negative_int(
    parser: argparse.ArgumentParser,
    *,
    cli_value: int | None,
    env_name: str,
    default: int,
) -> int:
    if cli_value is not None:
        return cli_value
    try:
        return parse_non_negative_int_env(env_name, default)
    except CoordinationError as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")


def _is_transport_closed_error(exc: BaseException) -> bool:
    """Walk exception tree (groups, __cause__, __context__) for transport errors."""
    stack: list[BaseException] = [exc]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, _TRANSPORT_CLOSE_EXCEPTIONS):
            return True
        if any(h in str(current).lower() for h in _TRANSPORT_CLOSE_HINTS):
            return True

        # Traverse ExceptionGroup-like .exceptions and chained causes
        nested = getattr(current, "exceptions", None)
        if isinstance(nested, (tuple, list)):
            stack.extend(e for e in nested if isinstance(e, BaseException))
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                stack.append(linked)

    return False


def _silence_stdout() -> None:
    with suppress(Exception):
        sys.stdout.flush()

    with suppress(Exception):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Lean LSP MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport method for the server. Default is 'stdio'.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for transport",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Host port for transport",
    )
    parser.add_argument(
        "--lean-project-path",
        type=str,
        help=("Path to a Lean project root or to a file/dir inside it."),
    )
    parser.add_argument(
        "--loogle-local",
        action="store_true",
        help="Enable local loogle (auto-installs on first run, ~5-10 min). "
        "Avoids rate limits and network dependencies.",
    )
    parser.add_argument(
        "--loogle-cache-dir",
        type=str,
        help="Override loogle cache location (default: ~/.cache/lean-lsp-mcp/loogle)",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Enable fast REPL-based multi-attempt (~5x faster). Requires Lean REPL.",
    )
    parser.add_argument(
        "--repl-timeout",
        type=int,
        help="REPL command timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--coordination",
        type=str,
        choices=[COORDINATION_MODE_DIRECT, COORDINATION_MODE_BROKER],
        default=None,
        help=(
            "Cross-process coordination mode. "
            "'direct' keeps the legacy per-process behavior; "
            "'broker' enables broker-managed worker leases."
        ),
    )
    parser.add_argument(
        "--coordination-dir",
        type=str,
        default=None,
        help="Directory used by broker coordination mode (socket + state).",
    )
    parser.add_argument(
        "--max-lineage-depth",
        type=int,
        default=None,
        help="Maximum allowed nested lineage depth for spawned Lean MCP servers.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum coordinated Lean workers across broker-managed instances.",
    )
    args = parser.parse_args()

    coordination_mode = _resolve_coordination_mode(parser, args.coordination)
    coordination_dir = str(
        Path(
            args.coordination_dir
            or os.environ.get(ENV_COORDINATION_DIR, str(default_coordination_dir()))
        )
        .expanduser()
        .resolve()
    )
    max_lineage_depth = _resolve_non_negative_int(
        parser,
        cli_value=args.max_lineage_depth,
        env_name=ENV_MAX_LINEAGE_DEPTH,
        default=DEFAULT_MAX_LINEAGE_DEPTH,
    )
    max_workers = _resolve_non_negative_int(
        parser,
        cli_value=args.max_workers,
        env_name=ENV_MAX_WORKERS,
        default=DEFAULT_MAX_WORKERS,
    )

    if max_lineage_depth < 0:
        parser.error("--max-lineage-depth must be >= 0")
    if max_workers <= 0:
        parser.error("--max-workers must be > 0")
    if coordination_mode == COORDINATION_MODE_BROKER and not _broker_coordination_supported():
        parser.error("Broker coordination mode is currently supported on Unix only.")

    # Set env vars from CLI args (CLI takes precedence over env vars)
    if args.lean_project_path:
        project_path = infer_project_path(args.lean_project_path)
        if project_path is None:
            parser.error(f"No lean-toolchain found for: {args.lean_project_path}")
        os.environ["LEAN_PROJECT_PATH"] = str(project_path)
    if args.loogle_local:
        os.environ["LEAN_LOOGLE_LOCAL"] = "true"
    if args.loogle_cache_dir:
        os.environ["LEAN_LOOGLE_CACHE_DIR"] = args.loogle_cache_dir
    if args.repl:
        os.environ["LEAN_REPL"] = "true"
    if args.repl_timeout:
        os.environ["LEAN_REPL_TIMEOUT"] = str(args.repl_timeout)
    os.environ["LEAN_LSP_MCP_ACTIVE_TRANSPORT"] = args.transport
    os.environ[ENV_COORDINATION_MODE] = coordination_mode
    os.environ[ENV_COORDINATION_DIR] = coordination_dir
    os.environ[ENV_MAX_LINEAGE_DEPTH] = str(max_lineage_depth)
    os.environ[ENV_MAX_WORKERS] = str(max_workers)

    current_pid = str(os.getpid())
    instance_id = os.environ.get(ENV_INSTANCE_ID, "").strip() or uuid.uuid4().hex
    os.environ[ENV_INSTANCE_ID] = instance_id

    # If main() is called repeatedly in the same process (e.g. unit tests),
    # avoid re-incrementing depth. Child processes still increment correctly.
    if os.environ.get("LEAN_LSP_MCP_INSTANCE_PID") == current_pid:
        lineage_root = os.environ.get(ENV_LINEAGE_ROOT, "").strip() or instance_id
        try:
            lineage_depth = int(os.environ.get(ENV_LINEAGE_DEPTH, "0") or "0")
        except ValueError:
            lineage_depth = 0
    else:
        try:
            lineage_root, lineage_depth = derive_lineage(instance_id)
        except CoordinationError as exc:
            parser.error(str(exc))

    if lineage_depth > max_lineage_depth:
        parser.error(
            "Lineage depth "
            f"{lineage_depth} exceeds max allowed depth {max_lineage_depth}."
        )

    os.environ["LEAN_LSP_MCP_INSTANCE_PID"] = current_pid
    os.environ[ENV_LINEAGE_ROOT] = lineage_root
    os.environ[ENV_LINEAGE_DEPTH] = str(lineage_depth)

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.transport == "stdio" and _is_transport_closed_error(exc):
            _silence_stdout()
            return 0
        raise
    return 0
