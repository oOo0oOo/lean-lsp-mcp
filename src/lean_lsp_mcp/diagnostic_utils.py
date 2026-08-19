"""Pure conversions for Lean goals and diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

from leanclient.aio import GoalResult

from lean_lsp_mcp.models import (
    DiagnosticMessage,
    DiagnosticsResult,
    GoalContextEntry,
    StructuredGoal,
)
from lean_lsp_mcp.utils import (
    LeanToolError,
    extract_failed_dependency_paths,
    is_build_stderr,
)

DIAGNOSTIC_SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}
LEAN_DIAGNOSTIC_TAG = {1: "unsolvedGoals", 2: "goalsAccomplished"}


def _lean_tags(diagnostic: dict) -> list[str] | None:
    tags = diagnostic.get("leanTags")
    if not tags:
        return None
    return [LEAN_DIAGNOSTIC_TAG.get(tag, str(tag)) for tag in tags]


def _diagnostic_message(diagnostic: dict) -> DiagnosticMessage | None:
    diagnostic_range = diagnostic.get("fullRange", diagnostic.get("range"))
    if diagnostic_range is None:
        return None
    severity = diagnostic.get("severity", 1)
    start = diagnostic_range["start"]
    return DiagnosticMessage(
        severity=DIAGNOSTIC_SEVERITY.get(severity, f"unknown({severity})"),
        message=diagnostic.get("message", ""),
        line=start["line"] + 1,
        column=start["character"] + 1,
        lean_tags=_lean_tags(diagnostic),
    )


def to_diagnostic_messages(diagnostics: Iterable[dict]) -> list[DiagnosticMessage]:
    """Convert LSP diagnostics to the public diagnostic model."""
    return [
        message
        for diagnostic in diagnostics
        if (message := _diagnostic_message(diagnostic)) is not None
    ]


def process_diagnostics(
    diagnostics: list[dict],
    build_success: bool,
    severity: str | None = None,
    timed_out: bool = False,
    partial: bool = False,
    processing_lines: list[list[int]] | None = None,
) -> DiagnosticsResult:
    """Convert diagnostics, filtering severity and extracting build failures."""
    items: list[DiagnosticMessage] = []
    failed_dependencies: list[str] = []

    for diagnostic in diagnostics:
        message = _diagnostic_message(diagnostic)
        if message is None:
            continue
        if (
            message.line == 1
            and message.column == 1
            and is_build_stderr(message.message)
        ):
            failed_dependencies = extract_failed_dependency_paths(message.message)
            continue
        if severity is None or message.severity == severity:
            items.append(message)

    if partial:
        return DiagnosticsResult(
            partial=True,
            still_elaborating_lines=processing_lines,
            success=False,
            timed_out=True,
            items=items,
            failed_dependencies=failed_dependencies,
        )

    if (not build_success or timed_out) and not items:
        reason = "diagnostics_timed_out" if timed_out else "diagnostics_unavailable"
        items.append(
            DiagnosticMessage(
                severity="error",
                message=f"{reason}: Lean did not finish; the file is not known clean.",
                line=1,
                column=1,
            )
        )
    return DiagnosticsResult(
        success=build_success,
        timed_out=timed_out,
        items=items,
        failed_dependencies=failed_dependencies,
    )


def goal_to_structured(goal: str) -> StructuredGoal:
    goal = (goal or "").strip()
    if not goal or goal.lower() == "no goals":
        return StructuredGoal(context=[], goal=None, status="complete", pretty=goal)
    if "⊢" not in goal:
        return StructuredGoal(context=[], goal=goal, status="unknown", pretty=goal)

    context_text, target = goal.split("⊢", 1)
    context: list[GoalContextEntry] = []
    current_name: str | None = None
    current_type_lines: list[str] = []

    def flush() -> None:
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

    for line in context_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped and not line.startswith(" "):
            flush()
            name, value_type = stripped.split(":", 1)
            current_name = name.strip()
            current_type_lines = [value_type.strip()]
        elif current_name is not None:
            current_type_lines.append(stripped)
    flush()

    return StructuredGoal(
        context=context,
        goal=target.strip(),
        status="open",
        pretty=goal,
    )


def goal_strings(goal: GoalResult) -> list[str]:
    """Flatten a GoalResult to the legacy list-of-goals output shape."""
    return list(goal.goals) if goal.status == "goals" else []


def get_line_context(lines: list[str], line: int) -> str:
    if line < 1 or line > len(lines):
        raise LeanToolError(f"Line {line} out of range (file has {len(lines)} lines)")
    return lines[line - 1]


def resolve_multi_attempt_column(line_context: str, column: int | None) -> int:
    if column is None:
        return next((i for i, char in enumerate(line_context) if not char.isspace()), 0)
    if column > len(line_context) + 1:
        raise LeanToolError(
            f"Column {column} out of range for line of length {len(line_context)}"
        )
    return column - 1


def diagnostic_identity(diagnostic: dict) -> tuple:
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


def shift_baseline_keys(
    baseline_keys: set[tuple], edit_start_line: int, line_delta: int
) -> set[tuple]:
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
        shifted.add(
            (
                start_line + line_delta,
                start_char,
                end_line + line_delta if end_line is not None else end_line,
                end_char,
                severity,
                code,
                source,
                message,
            )
        )
    return shifted


def filter_diagnostics_by_line_range(
    diagnostics: Iterable[dict], start_line: int, end_line: int
) -> list[dict]:
    matches = []
    for diagnostic in diagnostics:
        diagnostic_range = diagnostic.get("range") or diagnostic.get("fullRange")
        if not diagnostic_range:
            continue
        start = diagnostic_range.get("start", {}).get("line")
        end = diagnostic_range.get("end", {}).get("line")
        if start is None or end is None:
            continue
        if end >= start_line and start <= end_line:
            matches.append(diagnostic)
    return matches
