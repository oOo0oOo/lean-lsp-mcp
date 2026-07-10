"""Analysis tools: multi-attempt, run-code, verify, minimal-hypotheses, profiling.

All temporary-code checks run on pre-warmed scratch documents (virtual files
that never touch disk); the user's files and open documents are never edited.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Dict, List, Optional

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import server
from lean_lsp_mcp.file_utils import read_lean_source_utf8
from lean_lsp_mcp.client_utils import (
    get_path_policy,
    get_scratch_pool,
    infer_project_path,
    open_synced,
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

    # Priority 2: scratch-pool trials (parallel, user's document untouched)
    return await server._multi_attempt_lsp(ctx, file_path, line, column, snippets)


@server.mcp.tool(
    "lean_run_code",
    annotations=ToolAnnotations(
        title="Run Code",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def run_code(
    ctx: Context,
    code: Annotated[str, Field(description="Self-contained Lean code with imports")],
) -> RunResult:
    """Run a code snippet and return diagnostics. Must include all imports."""
    lifespan_context = ctx.request_context.lifespan_context
    if lifespan_context.lean_project_path is None:
        raise server.LeanToolError(
            "No valid Lean project path found. Run another tool first to set it up."
        )

    await startup_client(ctx)
    pool = get_scratch_pool(ctx)
    trial = await pool.run_text(code if code.endswith("\n") else code + "\n")

    diagnostics = server._to_diagnostic_messages(trial.diagnostics.items)
    has_errors = any(d.severity == "error" for d in diagnostics)

    return RunResult(
        success=not has_errors and not trial.diagnostics.fatal_error,
        timed_out=False,
        diagnostics=diagnostics,
    )


@server.mcp.tool(
    "lean_verify",
    annotations=ToolAnnotations(
        title="Verify Theorem",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def verify_theorem(
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
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    try:
        policy = get_path_policy(ctx)
        abs_path = policy.validate_path(server.resolve_file_path(ctx, file_path))
    except (FileNotFoundError, ValueError) as exc:
        raise server.LeanToolError(str(exc)) from exc

    try:
        original_content = read_lean_source_utf8(abs_path)
    except ValueError as exc:
        raise server.LeanToolError(str(exc)) from exc

    # Run the axiom check on a scratch copy; the real document is untouched.
    text = original_content.rstrip("\n") + f"\n\n#print axioms _root_.{theorem_name}\n"
    snippet_line = text.count("\n") - 1

    pool = get_scratch_pool(ctx)
    from leanclient.aio import LeanRequestTimeout

    try:
        trial = await pool.run_text(text)
    except LeanRequestTimeout as exc:
        raise server.LeanToolError(
            f"Axiom check timed out before `#print axioms` produced output: {exc}"
        ) from exc

    appended_diags = [
        d
        for d in trial.diagnostics.items
        if (d.get("fullRange") or d.get("range") or {}).get("start", {}).get("line", -1)
        >= snippet_line
    ]

    if err := check_axiom_errors(appended_diags):
        raise server.LeanToolError(f"Axiom check failed: {err}")

    axioms = parse_axioms(appended_diags)
    if not axioms and not any(
        "does not depend on any axioms" in d.get("message", "") for d in appended_diags
    ):
        raise server.LeanToolError(
            "Axiom check produced no `#print axioms` output — result is "
            "inconclusive (elaboration may not have reached the check)."
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
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def minimal_hypotheses(
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
            description="Per-hypothesis elaboration timeout (seconds)",
            ge=5.0,
            le=300.0,
        ),
    ] = 60.0,
) -> MinimalHypothesesResult:
    """For each explicit `(h : T)` hypothesis of a theorem, drop it and re-elaborate
    a scratch copy of the file. Reports which hypotheses are load-bearing and which
    are actually unused. Skips implicit `{x : α}` and instance `[inst : C]` binders
    (those are usually inferable / always load-bearing). Does not rewrite the proof
    body — a body that names `h` will fail to elaborate without the binder, which
    is the truthful answer (load-bearing).

    Variants are checked in parallel on scratch documents; the file is never edited."""
    from lean_lsp_mcp.minimal_hypotheses import (
        drop_binder,
        explicit_hypotheses,
        find_theorem_binders,
        theorem_declared,
    )

    theorem_name = server._validate_theorem_name(theorem_name)
    rel_path = await server.setup_client_for_file(ctx, file_path)
    if not rel_path:
        server._raise_invalid_path(file_path)

    client = ctx.request_context.lifespan_context.client
    doc = await open_synced(ctx, rel_path)
    original_content = doc.text

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

    baseline = await client.diagnostics(rel_path, timeout=inactivity_timeout)
    baseline_keys = {_error_key(d) for d in baseline.items if d.get("severity") == 1}

    pool = get_scratch_pool(ctx)
    from leanclient.aio import LeanClientError, LeanRequestTimeout

    async def probe(binder: str, start: int, end: int) -> HypothesisVerdict:
        modified = drop_binder(original_content, start, end)
        try:
            trial = await pool.run_text(modified, timeout=inactivity_timeout)
        except LeanRequestTimeout:
            return HypothesisVerdict(
                binder=binder.strip(),
                status=HypothesisStatus.error,
                detail=f"elaboration timed out after {inactivity_timeout}s",
            )
        except LeanClientError as exc:
            return HypothesisVerdict(
                binder=binder.strip(),
                status=HypothesisStatus.error,
                detail=str(exc)[:200],
            )

        new_errors = [
            d
            for d in trial.diagnostics.items
            if d.get("severity") == 1 and _error_key(d) not in baseline_keys
        ]
        if new_errors:
            return HypothesisVerdict(
                binder=binder.strip(),
                status=HypothesisStatus.load_bearing,
                breaks=server._to_diagnostic_messages(new_errors),
            )
        return HypothesisVerdict(
            binder=binder.strip(),
            status=HypothesisStatus.removable,
        )

    verdicts = list(
        await asyncio.gather(
            *(probe(binder, start, end) for binder, start, end in explicit)
        )
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
