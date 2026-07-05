"""Search tools: local ripgrep search plus the remote search backends."""

from __future__ import annotations

import asyncio
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Annotated, List, Literal, Optional

import orjson
from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import Field

from lean_lsp_mcp import config, server
from lean_lsp_mcp.client_utils import bind_lean_project_path, build_lean_path_policy
from lean_lsp_mcp.loogle import loogle_remote
from lean_lsp_mcp.models import (
    LeanFinderResult,
    LeanFinderResults,
    LeanSearchResult,
    LeanSearchResults,
    LocalSearchResult,
    LocalSearchResults,
    LoogleResult,
    LoogleResults,
    PremiseResult,
    PremiseResults,
    StateSearchResult,
    StateSearchResults,
)


class LocalSearchError(Exception):
    pass


def _get_goal_for_remote_search(
    ctx: Context, file_path: str, line: int, column: int
) -> dict | None:
    try:
        with server.lsp_client_for_file(ctx, file_path) as lsp:
            return lsp.client.get_goal(lsp.rel_path, line - 1, column - 1)
    except server.InvalidLeanFilePathError:
        server._raise_invalid_path(file_path)


@server.mcp.tool(
    "lean_local_search",
    annotations=ToolAnnotations(
        title="Local Search",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def local_search(
    ctx: Context,
    query: Annotated[str, Field(description="Declaration name or prefix")],
    limit: Annotated[int, Field(description="Max matches", ge=1)] = 10,
    project_root: Annotated[
        Optional[str], Field(description="Project root (inferred if omitted)")
    ] = None,
) -> LocalSearchResults:
    """Fast local search to verify declarations exist. Use BEFORE trying a lemma name."""
    if not server._RG_AVAILABLE:
        raise LocalSearchError(server._RG_MESSAGE)

    lifespan = ctx.request_context.lifespan_context
    stored_root = lifespan.lean_project_path

    if project_root:
        try:
            root_path = Path(project_root).expanduser()
            if not root_path.is_absolute() and stored_root is not None:
                root_path = stored_root / root_path
            previous_root = stored_root
            resolved_root = bind_lean_project_path(ctx, root_path)
            if previous_root is not None and previous_root != resolved_root:
                await server._close_repl_for_project_switch(lifespan)
        except (OSError, ValueError) as exc:
            raise LocalSearchError(f"Invalid project root '{project_root}': {exc}")
    else:
        resolved_root = stored_root

    if resolved_root is None:
        raise LocalSearchError(
            "Lean project path not set. Call a file-based tool first."
        )

    try:
        policy = build_lean_path_policy(resolved_root)
        raw_results = await asyncio.to_thread(
            server.lean_local_search,
            query=query.strip(),
            limit=limit,
            project_root=policy.project_root,
            path_policy=policy,
        )
        results = [
            LocalSearchResult(name=r["name"], kind=r["kind"], file=r["file"])
            for r in raw_results
        ]
        return LocalSearchResults(items=results)
    except RuntimeError as exc:
        raise LocalSearchError(f"Search failed: {exc}")


@server.mcp.tool(
    "lean_leansearch",
    annotations=ToolAnnotations(
        title="LeanSearch",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
# leansearch.net raised capacity to ~40 req/s (per-IP ~120 req/min) and asked
# us to lift our previously very conservative 3 req/30s. This client-side
# throttle mainly guards against runaway loops; the server enforces its own
# per-IP limit, which the maintainers adjust dynamically as capacity allows.
@server.rate_limited("leansearch", max_requests=90, per_seconds=30)
async def leansearch(
    ctx: Context,
    query: Annotated[str, Field(description="Natural language or Lean term query")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> LeanSearchResults:
    """Search Mathlib via leansearch.net using natural language.

    Examples: "sum of two even numbers is even", "Cauchy-Schwarz inequality",
    "{f : A → B} (hf : Injective f) : ∃ g, LeftInverse g f"
    """
    headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
    payload = orjson.dumps({"num_results": str(num_results), "query": [query]})

    req = urllib.request.Request(
        "https://leansearch.net/search",
        data=payload,
        headers=headers,
        method="POST",
    )

    await server._safe_report_progress(
        ctx, progress=1, total=10, message="Awaiting response from leansearch.net"
    )
    results = await server._urlopen_json(req, timeout=10)

    if not results or not results[0]:
        return LeanSearchResults(items=[])

    raw_results = [r["result"] for r in results[0][:num_results]]
    items = [
        LeanSearchResult(
            name=".".join(r["name"]),
            module_name=".".join(r["module_name"]),
            kind=r.get("kind"),
            type=r.get("type"),
        )
        for r in raw_results
    ]
    return LeanSearchResults(items=items)


@server.mcp.tool(
    "lean_loogle",
    annotations=ToolAnnotations(
        title="Loogle",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def loogle(
    ctx: Context,
    query: Annotated[
        str, Field(description="Type pattern, constant, or name substring")
    ],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 8,
) -> LoogleResults:
    """Search Mathlib by type signature via loogle.lean-lang.org.

    Examples: `Real.sin`, `"comm"`, `(?a → ?b) → List ?a → List ?b`,
    `_ * (_ ^ _)`, `|- _ < _ → _ + 1 < _ + 1`
    """
    app_ctx = ctx.request_context.lifespan_context

    # Try local loogle first if available (no rate limiting)
    if app_ctx.loogle_local_available and app_ctx.loogle_manager:
        # Update project path if it changed (adds new library paths)
        if app_ctx.lean_project_path != app_ctx.loogle_manager.project_path:
            if app_ctx.loogle_manager.set_project_path(app_ctx.lean_project_path):
                # Restart to pick up new paths
                await app_ctx.loogle_manager.stop()
        try:
            results = await app_ctx.loogle_manager.query(query, num_results)
            if not results:
                return LoogleResults(items=[])
            items = [
                LoogleResult(
                    name=r.get("name", ""),
                    type=r.get("type", ""),
                    module=r.get("module", ""),
                )
                for r in results
            ]
            return LoogleResults(items=items)
        except Exception as e:
            server.logger.warning(f"Local loogle failed: {e}, falling back to remote")

    # Fall back to remote. Rate limit only the default public instance; a
    # custom LOOGLE_URL (self-hosted backend) is not rate limited.
    if not server._custom_backend("LOOGLE_URL", config.DEFAULT_LOOGLE_URL):
        rate_limit = app_ctx.rate_limit["loogle"]
        now = int(time.time())
        rate_limit[:] = [t for t in rate_limit if now - t < 30]
        if len(rate_limit) >= 3:
            raise server.LeanToolError(
                "Rate limit exceeded: 3 requests per 30s. Use --loogle-local to avoid limits."
            )
        rate_limit.append(now)

    await server._safe_report_progress(
        ctx,
        progress=1,
        total=10,
        message="Awaiting response from loogle.lean-lang.org",
    )
    result = await asyncio.to_thread(loogle_remote, query, num_results)
    if isinstance(result, str):
        raise server.LeanToolError(result)  # Error message from remote
    return LoogleResults(items=result)


@server.mcp.tool(
    "lean_leanfinder",
    annotations=ToolAnnotations(
        title="Lean Finder",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@server.rate_limited("leanfinder", max_requests=10, per_seconds=30)
async def leanfinder(
    ctx: Context,
    query: Annotated[str, Field(description="Mathematical concept or proof state")],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
    version: Annotated[
        Literal["v4.19.0", "v4.24.0", "v4.28.0"],
        Field(description="Mathlib version index to search"),
    ] = "v4.28.0",
) -> LeanFinderResults:
    """Semantic search by mathematical meaning via Lean Finder.

    Examples: "commutativity of addition on natural numbers",
    "I have h : n < m and need n + 1 < m + 1", proof state text.

    The `version` argument selects which mathlib snapshot to query
    (v4.19.0, v4.24.0, or v4.28.0). Default: v4.28.0.
    """
    headers = {"User-Agent": "lean-lsp-mcp/0.1", "Content-Type": "application/json"}
    request_url = "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
    payload = orjson.dumps(
        {"inputs": query, "top_k": int(num_results), "version": version}
    )
    req = urllib.request.Request(
        request_url, data=payload, headers=headers, method="POST"
    )

    await server._safe_report_progress(
        ctx,
        progress=1,
        total=10,
        message="Awaiting response from Lean Finder (Hugging Face)",
    )
    data = await server._urlopen_json(req, timeout=10)
    if isinstance(data, dict) and "error" in data:
        raise server.LeanToolError(str(data["error"]))

    results: List[LeanFinderResult] = []
    for r in data.get("results", []):
        results.append(
            LeanFinderResult(
                formal_name=r.get("formal_name", ""),
                informal_name=r.get("informal_name", ""),
                kind=r.get("kind", ""),
                type=r.get("type", ""),
                informal_description=r.get("informal_description", ""),
                path=r.get("path", "").replace("/", "."),
            )
        )

    return LeanFinderResults(items=results)


@server.mcp.tool(
    "lean_state_search",
    annotations=ToolAnnotations(
        title="State Search",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@server.rate_limited(
    "lean_state_search",
    max_requests=6,
    per_seconds=30,
    bypass=lambda: server._custom_backend(
        "LEAN_STATE_SEARCH_URL", config.DEFAULT_STATE_SEARCH_URL
    ),
)
async def state_search(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 5,
) -> StateSearchResults:
    """Find lemmas to close the goal at a position. Searches premise-search.com."""
    goal = await asyncio.to_thread(
        _get_goal_for_remote_search, ctx, file_path, line, column
    )

    if not goal or not goal.get("goals"):
        raise server.LeanToolError(
            f"No goals found at line {line}, column {column}. Try a different position or check if the proof is complete."
        )

    goal_str = urllib.parse.quote(goal["goals"][0])

    url = config.state_search_url()
    req = urllib.request.Request(
        f"{url}/api/search?query={goal_str}&results={num_results}&rev=v4.22.0",
        headers={"User-Agent": "lean-lsp-mcp/0.1"},
        method="GET",
    )

    await server._safe_report_progress(
        ctx, progress=1, total=10, message=f"Awaiting response from {url}"
    )
    results = await server._urlopen_json(req, timeout=10)

    items = [StateSearchResult(name=r["name"]) for r in results]
    return StateSearchResults(items=items)


@server.mcp.tool(
    "lean_hammer_premise",
    annotations=ToolAnnotations(
        title="Hammer Premises",
        readOnlyHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
@server.rate_limited(
    "hammer_premise",
    max_requests=6,
    per_seconds=30,
    bypass=lambda: server._custom_backend("LEAN_HAMMER_URL", config.DEFAULT_HAMMER_URL),
)
async def hammer_premise(
    ctx: Context,
    file_path: Annotated[
        str, Field(description="Absolute or project-root-relative path to Lean file")
    ],
    line: Annotated[int, Field(description="Line number (1-indexed)", ge=1)],
    column: Annotated[int, Field(description="Column number (1-indexed)", ge=1)],
    num_results: Annotated[int, Field(description="Max results", ge=1)] = 32,
) -> PremiseResults:
    """Get premise suggestions for automation tactics at a goal position.

    Returns lemma names to try with `simp only [...]`, `aesop`, or as hints.
    """
    goal = await asyncio.to_thread(
        _get_goal_for_remote_search, ctx, file_path, line, column
    )

    if not goal or not goal.get("goals"):
        raise server.LeanToolError(
            f"No goals found at line {line}, column {column}. Try a different position or check if the proof is complete."
        )

    data = {
        "state": goal["goals"][0],
        "new_premises": [],
        "k": num_results,
    }

    url = config.hammer_url()
    req = urllib.request.Request(
        url + "/retrieve",
        headers={
            "User-Agent": "lean-lsp-mcp/0.1",
            "Content-Type": "application/json",
        },
        method="POST",
        data=orjson.dumps(data),
    )

    await server._safe_report_progress(
        ctx, progress=1, total=10, message=f"Awaiting response from {url}"
    )
    results = await server._urlopen_json(req, timeout=10)

    items = [PremiseResult(name=r["name"]) for r in results]
    return PremiseResults(items=items)
