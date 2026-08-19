"""Execution helpers shared by the multi-attempt and run-code tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leanclient.aio import LeanClientError
from mcp.server.mcpserver.utilities.logging import get_logger

from lean_lsp_mcp.client_utils import (
    get_client,
    get_scratch_pool,
    infer_project_path,
    open_synced,
    require_client_for_file,
    resolve_file_path,
)
from lean_lsp_mcp.diagnostic_utils import (
    diagnostic_identity,
    filter_diagnostics_by_line_range,
    get_line_context,
    goal_strings,
    resolve_multi_attempt_column,
    shift_baseline_keys,
    to_diagnostic_messages,
)
from lean_lsp_mcp.file_utils import build_lean_path_policy, get_file_contents
from lean_lsp_mcp.models import (
    AttemptResult,
    DiagnosticMessage,
    MultiAttemptResult,
    RunResult,
)
from lean_lsp_mcp.repl import ReplProcessError, ReplRunResult

logger = get_logger(__name__)


async def close_repl_for_project_switch(app_ctx: Any) -> None:
    repl = app_ctx.repl
    app_ctx.repl_enabled = False
    if repl is None:
        return
    app_ctx.repl = None
    try:
        await repl.close()
    except Exception:
        logger.exception("REPL close failed during project switch")


def build_attempt_text(
    lines: list[str], line_context: str, target_column: int, snippet: str, line: int
) -> tuple[str, str, int, int, int]:
    """Build a trial document and return its text and cursor metadata."""
    snippet_str = snippet.rstrip("\n")
    snippet_lines = snippet_str.split("\n") if snippet_str else [""]
    indent = line_context[:target_column]
    payload_lines = [
        line_context[:target_column] + snippet_lines[0],
        *[f"{indent}{part}" for part in snippet_lines[1:]],
    ]

    end_line = min(line - 1 + len(snippet_lines), len(lines))
    line_delta = len(payload_lines) - (end_line - (line - 1))
    text = "\n".join(lines[: line - 1] + payload_lines + lines[end_line:]) + "\n"
    goal_line = line - 1 + len(payload_lines) - 1
    goal_column = len(payload_lines[-1])
    return snippet_str, text, goal_line, goal_column, line_delta


def _attempt_results(
    snippets: list[str], repl_results: list[Any]
) -> MultiAttemptResult:
    results = []
    for snippet, result in zip(snippets, repl_results):
        diagnostics = [
            DiagnosticMessage(
                severity=message.get("severity", "info"),
                message=message.get("data", ""),
                line=message.get("pos", {}).get("line", 0),
                column=message.get("pos", {}).get("column", 0),
            )
            for message in (result.messages or [])
        ]
        if result.error:
            diagnostics.append(
                DiagnosticMessage(
                    severity="error", message=result.error, line=0, column=0
                )
            )
        if result.proof_status and result.proof_status != "Completed":
            diagnostics.append(
                DiagnosticMessage(
                    severity="warning",
                    message=f"REPL proof status: {result.proof_status}",
                    line=0,
                    column=0,
                )
            )
        results.append(
            AttemptResult(
                snippet=snippet.rstrip("\n"),
                goals=result.goals or [],
                diagnostics=diagnostics,
                proof_status=result.proof_status,
            )
        )
    return MultiAttemptResult(items=results)


async def multi_attempt_repl(
    ctx: Any,
    file_path: str,
    line: int,
    column: int | None = None,
    snippets: list[str] | None = None,
) -> MultiAttemptResult | None:
    """Try tactics using the session REPL fast path when it is applicable."""
    app_ctx = ctx.request_context.lifespan_context
    snippets = snippets or []
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
        resolved_path = build_lean_path_policy(project_path).validate_path(
            resolved_path
        )
        if Path(app_ctx.repl.project_dir).resolve(strict=False) != project_path:
            await close_repl_for_project_switch(app_ctx)
            return None
        content = get_file_contents(str(resolved_path))
        if content is None:
            return None
        lines = content.splitlines()
        if line > len(lines):
            return None

        repl_results = await app_ctx.repl.run_snippets(
            "\n".join(lines[: line - 1]), snippets
        )
        return _attempt_results(snippets, repl_results)
    except ReplProcessError as error:
        await disable_unhealthy_repl(app_ctx, "multi_attempt", error)
        return None
    except Exception as error:
        logger.debug("REPL multi_attempt failed: %s", error)
        return None


def repl_run_diagnostics(result: ReplRunResult) -> list[DiagnosticMessage]:
    """Convert Lean REPL messages to the existing run-code diagnostic shape."""
    diagnostics = []
    for message in result.messages:
        position = message.get("pos") or {}
        severity = message.get("severity", "info")
        if severity == "information":
            severity = "info"
        diagnostics.append(
            DiagnosticMessage(
                severity=severity,
                message=str(message.get("data", "")),
                line=max(1, int(position.get("line", 1)) + result.line_offset),
                column=max(1, int(position.get("column", 0)) + 1),
            )
        )
    if result.error:
        diagnostics.append(
            DiagnosticMessage(
                severity="error",
                message=str(result.error),
                line=max(1, result.line_offset + 1),
                column=1,
            )
        )
    return diagnostics


async def disable_unhealthy_repl(
    app_ctx: Any, operation: str, error: ReplProcessError
) -> None:
    """Disable a failed fast lane once instead of retrying every call."""
    app_ctx.repl_enabled = False
    logger.warning(
        "REPL %s fast path is unhealthy; disabling it for this session and "
        "falling back to LSP: %s",
        operation,
        error,
    )
    if app_ctx.repl is not None:
        try:
            await app_ctx.repl.close()
        except Exception:
            logger.exception("REPL close failed after %s failure", operation)


async def run_code_repl(ctx: Any, code: str) -> RunResult | None:
    """Run code through the session REPL when its import cache is healthy."""
    app_ctx = ctx.request_context.lifespan_context
    if not app_ctx.repl_enabled or app_ctx.repl is None:
        return None

    project_path = app_ctx.lean_project_path
    if project_path is None:
        return None
    if Path(app_ctx.repl.project_dir).resolve(strict=False) != Path(
        project_path
    ).resolve(strict=False):
        await close_repl_for_project_switch(app_ctx)
        return None

    try:
        result = await app_ctx.repl.run_code(code)
    except ReplProcessError as error:
        await disable_unhealthy_repl(app_ctx, "run_code", error)
        return None
    except Exception as error:
        logger.debug("REPL run_code failed: %s", error)
        return None

    diagnostics = repl_run_diagnostics(result)
    return RunResult(
        success=not any(diagnostic.severity == "error" for diagnostic in diagnostics),
        timed_out=False,
        diagnostics=diagnostics,
    )


async def multi_attempt_lsp(
    ctx: Any,
    file_path: str,
    line: int,
    column: int | None = None,
    snippets: list[str] | None = None,
) -> MultiAttemptResult:
    """Try tactics on scratch documents without editing the user's file."""
    snippets = snippets or []
    relative_path = await require_client_for_file(ctx, file_path)
    client = get_client(ctx)
    document = await open_synced(ctx, relative_path)
    lines = document.text.splitlines() if document.text is not None else []
    line_context = get_line_context(lines, line)
    target_column = resolve_multi_attempt_column(line_context, column)

    try:
        baseline_report = await client.diagnostics(relative_path)
        baseline_keys = {
            diagnostic_identity(diagnostic) for diagnostic in baseline_report.items
        }
    except LeanClientError:
        logger.warning(
            "multi_attempt_lsp: baseline diagnostics unavailable; "
            "set-diff disabled and results limited to the local line range",
            exc_info=True,
        )
        baseline_keys = None

    prepared = [
        build_attempt_text(lines, line_context, target_column, snippet, line)
        for snippet in snippets
    ]
    trials = await get_scratch_pool(ctx).run_texts(
        [text for _, text, _, _, _ in prepared],
        want_goal_at=[
            (goal_line, goal_column) for _, _, goal_line, goal_column, _ in prepared
        ],
    )

    results = []
    for (snippet, _, goal_line, _, line_delta), trial in zip(prepared, trials):
        diagnostics = trial.diagnostics.items
        local_diagnostics = filter_diagnostics_by_line_range(
            diagnostics, line - 1, goal_line
        )
        local_ids = {id(diagnostic) for diagnostic in local_diagnostics}
        if baseline_keys is None:
            extra_diagnostics = []
        else:
            shifted_keys = shift_baseline_keys(
                baseline_keys, edit_start_line=line - 1, line_delta=line_delta
            )
            extra_diagnostics = [
                diagnostic
                for diagnostic in diagnostics
                if id(diagnostic) not in local_ids
                and diagnostic_identity(diagnostic) not in shifted_keys
            ]
        results.append(
            AttemptResult(
                snippet=snippet,
                goals=goal_strings(trial.goal) if trial.goal else [],
                diagnostics=to_diagnostic_messages(
                    local_diagnostics + extra_diagnostics
                ),
                timed_out=False,
            )
        )
    return MultiAttemptResult(items=results)
