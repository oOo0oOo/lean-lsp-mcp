"""Analysis tools: multi-attempt, run-code, verify, minimal-hypotheses, profiling."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from typing import Annotated, Dict, List, Optional

from leanclient import DocumentContentChange, LeanLSPClient
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.client_utils import (
    get_path_policy,
    infer_project_path,
    startup_client,
)
from lean_lsp_mcp.models import (
    HypothesisStatus,
    HypothesisVerdict,
    MinimalHypothesesResult,
    MultiAttemptResult,
    ProofProfileResult,
    RunResult,
    SourceWarning,
    VerifyResult,
)
from lean_lsp_mcp.utils import check_lsp_response


@server.mcp.tool(
    "lean_multi_attempt",
    annotations=ToolAnnotations(
        title="Multi-Attempt",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def multi_attempt(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    snippets: Annotated[
        List[str],
        Field(description="Tactics to try (3+ recommended)"),
    ],
    column: Annotated[
        Optional[int],
        Field(description="Column (1-indexed). Omit to target the tactic line", ge=1),
    ] = None,
) -> MultiAttemptResult:
    """Try multiple tactics without modifying file. Returns goal state for each."""
    # Priority 1: REPL
    result = await server._multi_attempt_repl(ctx, file_path, line, column, snippets)
    if result is not None:
        return result

    # Priority 2: LSP approach (fallback)
    return server._multi_attempt_lsp(ctx, file_path, line, column, snippets)


@server.mcp.tool(
    "lean_run_code",
    annotations=ToolAnnotations(
        title="Run Code",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def run_code(
    ctx: Context,
    code: Annotated[str, Field(description="Self-contained Lean code with imports")],
) -> RunResult:
    """Run a code snippet and return diagnostics. Must include all imports."""
    lifespan_context = ctx.request_context.lifespan_context
    lean_project_path = lifespan_context.lean_project_path
    if lean_project_path is None:
        raise server.LeanToolError(
            "No valid Lean project path found. Run another tool first to set it up."
        )

    # Use a unique snippet filename to avoid collisions under concurrency
    rel_path = f"_mcp_snippet_{uuid.uuid4().hex}.lean"
    abs_path = lean_project_path / rel_path

    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        raise server.LeanToolError(f"Error writing code snippet: {e}")

    client: LeanLSPClient | None = lifespan_context.client
    raw_diagnostics: Iterable[Dict] = []
    opened_file = False

    try:
        if client is None:
            startup_client(ctx)
            client = lifespan_context.client
            if client is None:
                raise server.LeanToolError(
                    "Failed to initialize Lean client for run_code."
                )

        client.open_file(rel_path)
        opened_file = True
        raw_diagnostics = client.get_diagnostics(rel_path, inactivity_timeout=15.0)
        check_lsp_response(raw_diagnostics, "get_diagnostics")
    finally:
        if opened_file:
            try:
                client.close_files([rel_path])
            except Exception as exc:
                server.logger.warning(
                    "Failed to close `%s` after run_code: %s", rel_path, exc
                )
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            server.logger.warning(
                "Failed to remove temporary Lean snippet `%s`: %s", abs_path, e
            )

    diagnostics = server._to_diagnostic_messages(raw_diagnostics)
    has_errors = any(d.severity == "error" for d in diagnostics)

    return RunResult(
        success=not has_errors,
        timed_out=getattr(raw_diagnostics, "timed_out", False),
        diagnostics=diagnostics,
    )


@server.mcp.tool(
    "lean_verify",
    annotations=ToolAnnotations(
        title="Verify Theorem",
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def verify_theorem(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    theorem_name: Annotated[
        str, Field(description="Fully qualified name (e.g. `Namespace.theorem`)")
    ],
    scan_source: Annotated[
        bool, Field(description="Scan source file for suspicious patterns")
    ] = True,
) -> VerifyResult:
    """Check theorem axioms + optional source scan. Only scans the given file, not imports."""
    from lean_lsp_mcp.verify import (
        check_axiom_errors,
        parse_axioms,
        scan_warnings,
    )

    theorem_name = server._validate_theorem_name(theorem_name)
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    try:
        policy = get_path_policy(ctx)
        abs_path = policy.validate_path(server.resolve_file_path(ctx, file_path))
    except (FileNotFoundError, ValueError) as exc:
        raise server.LeanToolError(str(exc)) from exc
    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)

    try:
        original_content = client.get_file_content(rel_path)
    except Exception:
        original_content = server.get_file_contents(abs_path)

    snippet = f"\n#print axioms _root_.{theorem_name}\n"
    original_lines = original_content.split("\n")
    appended_line = len(original_lines)  # 0-indexed line where snippet starts

    try:
        change = DocumentContentChange(
            snippet,
            (appended_line, 0),
            (appended_line, 0),
        )
        client.update_file(rel_path, [change])
        raw = client.get_diagnostics(
            rel_path, start_line=appended_line, inactivity_timeout=120.0
        )
        check_lsp_response(raw, "get_diagnostics")

        appended_diags = list(raw)

        if err := check_axiom_errors(appended_diags):
            raise server.LeanToolError(f"Axiom check failed: {err}")

        axioms = parse_axioms(appended_diags)
    finally:
        try:
            client.update_file_content(rel_path, original_content)
        except Exception as exc:
            server.logger.warning(
                "Failed to restore `%s` after verify: %s", rel_path, exc
            )
        try:
            client.open_file(rel_path, force_reopen=True)
        except Exception as exc:
            server.logger.warning(
                "Failed to force-reopen `%s` after verify: %s", rel_path, exc
            )

    w: list[SourceWarning] = []
    if scan_source:
        if server._RG_AVAILABLE:
            w = [
                SourceWarning(line=int(w["line"]), pattern=str(w["pattern"]))
                for w in scan_warnings(abs_path)
            ]
        else:
            w = [
                SourceWarning(
                    line=0, pattern="ripgrep (rg) not installed - warnings unavailable"
                )
            ]

    return VerifyResult(axioms=axioms, warnings=w)


@server.mcp.tool(
    "lean_minimal_hypotheses",
    annotations=ToolAnnotations(
        title="Minimal Hypotheses",
        readOnlyHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def minimal_hypotheses(
    ctx: Context,
    file_path: Annotated[str, Field(description="Absolute path to Lean file")],
    theorem_name: Annotated[
        str,
        Field(
            description=(
                "Theorem name. Either bare (e.g. `add_comm`) or fully qualified "
                "(e.g. `Namespace.add_comm`); only the trailing segment is used "
                "for source matching."
            ),
        ),
    ],
    inactivity_timeout: Annotated[
        float,
        Field(
            description="Per-hypothesis LSP elaboration timeout (seconds)",
            ge=5.0,
            le=300.0,
        ),
    ] = 60.0,
) -> MinimalHypothesesResult:
    """For each explicit `(h : T)` hypothesis of a theorem, drop it and re-elaborate
    the file via the LSP. Reports which hypotheses are load-bearing and which are
    actually unused. Skips implicit `{x : α}` and instance `[inst : C]` binders
    (those are usually inferable / always load-bearing). Does not rewrite the proof
    body — a body that names `h` will fail to elaborate without the binder, which
    is the truthful answer (load-bearing).

    Slow: each hypothesis triggers a re-elaboration capped at `inactivity_timeout`.
    The original file content is restored before the tool returns."""
    from lean_lsp_mcp.minimal_hypotheses import (
        drop_binder,
        explicit_hypotheses,
        find_theorem_binders,
        theorem_declared,
    )

    theorem_name = server._validate_theorem_name(theorem_name)
    rel_path = server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client: LeanLSPClient = ctx.request_context.lifespan_context.client
    client.open_file(rel_path)
    try:
        original_content = client.get_file_content(rel_path)
    except Exception:
        try:
            policy = get_path_policy(ctx)
            abs_path = policy.validate_path(server.resolve_file_path(ctx, file_path))
        except (FileNotFoundError, ValueError) as exc:
            raise server.LeanToolError(str(exc)) from exc
        original_content = server.get_file_contents(abs_path)

    if original_content is None:
        raise server.LeanToolError(f"Could not read content for {rel_path}")

    bare_name = theorem_name.split(".")[-1]
    if not theorem_declared(original_content, bare_name):
        raise server.LeanToolError(
            f"Could not find theorem '{theorem_name}' in {rel_path}."
        )

    binders = find_theorem_binders(original_content, bare_name)
    explicit = explicit_hypotheses(binders)
    skipped = len(binders) - len(explicit)

    # Theorem found but nothing explicit to probe (no binders, or only
    # implicit/instance binders): there is no hypothesis to minimize.
    if not explicit:
        return MinimalHypothesesResult(
            theorem_name=theorem_name,
            file=rel_path,
            verdicts=[],
            skipped_implicit=skipped,
        )

    def _error_key(diag: Dict) -> tuple[int, int, str]:
        """Stable identifier for a single LSP diagnostic — used to filter the
        set of *new* errors against the pre-modification baseline. Line and
        column may shift slightly if a multi-line binder is removed, but most
        binders are single-line so (line, column, message[:120]) is stable
        in practice.
        """
        r = diag.get("fullRange", diag.get("range")) or {}
        start = r.get("start", {})
        return (
            int(start.get("line", -1)),
            int(start.get("character", -1)),
            str(diag.get("message", ""))[:120],
        )

    baseline = client.get_diagnostics(rel_path, inactivity_timeout=inactivity_timeout)
    check_lsp_response(baseline, "get_diagnostics")
    baseline_keys = {
        _error_key(d) for d in list(baseline or []) if d.get("severity") == 1
    }

    verdicts: list[HypothesisVerdict] = []
    try:
        for binder, start, end in explicit:
            modified = drop_binder(original_content, start, end)
            try:
                client.update_file_content(rel_path, modified)
            except Exception as exc:
                verdicts.append(
                    HypothesisVerdict(
                        binder=binder.strip(),
                        status=HypothesisStatus.error,
                        detail=f"update_file_content failed: {exc}",
                    )
                )
                continue

            diag = client.get_diagnostics(
                rel_path, inactivity_timeout=inactivity_timeout
            )
            try:
                check_lsp_response(diag, "get_diagnostics")
            except Exception as exc:
                verdicts.append(
                    HypothesisVerdict(
                        binder=binder.strip(),
                        status=HypothesisStatus.error,
                        detail=str(exc)[:200],
                    )
                )
                continue

            if getattr(diag, "timed_out", False):
                verdicts.append(
                    HypothesisVerdict(
                        binder=binder.strip(),
                        status=HypothesisStatus.error,
                        detail=f"LSP elaboration timed out after {inactivity_timeout}s",
                    )
                )
                continue

            new_errors = [
                d
                for d in list(diag or [])
                if d.get("severity") == 1 and _error_key(d) not in baseline_keys
            ]
            if new_errors:
                breaks = server._to_diagnostic_messages(new_errors)
                verdicts.append(
                    HypothesisVerdict(
                        binder=binder.strip(),
                        status=HypothesisStatus.load_bearing,
                        breaks=breaks,
                    )
                )
            else:
                verdicts.append(
                    HypothesisVerdict(
                        binder=binder.strip(),
                        status=HypothesisStatus.removable,
                    )
                )
    finally:
        try:
            client.update_file_content(rel_path, original_content)
        except Exception as exc:
            server.logger.warning(
                "Failed to restore `%s` after minimal_hypotheses: %s", rel_path, exc
            )
        try:
            client.open_file(rel_path, force_reopen=True)
        except Exception as exc:
            server.logger.warning(
                "Failed to force-reopen `%s` after minimal_hypotheses: %s",
                rel_path,
                exc,
            )

    return MinimalHypothesesResult(
        theorem_name=theorem_name,
        file=rel_path,
        verdicts=verdicts,
        skipped_implicit=skipped,
    )


@server.mcp.tool(
    "lean_profile_proof",
    annotations=ToolAnnotations(
        title="Profile Proof",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def profile_proof(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[
        int, Field(description="Line where theorem starts (1-indexed)", ge=1)
    ],
    top_n: Annotated[
        int, Field(description="Number of slowest lines to return", ge=1)
    ] = 5,
    timeout: Annotated[float, Field(description="Max seconds to wait", ge=1)] = 60.0,
) -> ProofProfileResult:
    """Run `lean --profile` on a theorem. Returns per-line timing and categories. SLOW - avoid on theorems that already hit heartbeat limits."""
    from lean_lsp_mcp.profile_utils import profile_theorem

    file_path_obj = server.resolve_file_path(ctx, file_path)

    # Get project path
    lifespan = ctx.request_context.lifespan_context
    project_path = lifespan.lean_project_path

    if not project_path:
        project_path = infer_project_path(str(file_path_obj), ctx=ctx)
    if project_path is None:
        raise server.LeanToolError("Lean project not found")
    try:
        policy = get_path_policy(ctx, project_path)
        file_path_obj = policy.validate_path(file_path_obj)
    except ValueError as exc:
        raise server.LeanToolError(str(exc)) from exc

    try:
        return await profile_theorem(
            file_path=file_path_obj,
            theorem_line=line,
            project_path=project_path,
            timeout=timeout,
            top_n=top_n,
        )
    except (ValueError, TimeoutError) as e:
        raise server.LeanToolError(str(e)) from e
