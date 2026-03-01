"""Resolve Lean declaration names to file locations using ripgrep + LSP."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context

from lean_lsp_mcp.client_utils import setup_client_for_file
from lean_lsp_mcp.search_utils import lean_local_search
from lean_lsp_mcp.utils import LeanToolError, get_declaration_range

if TYPE_CHECKING:
    from leanclient import LeanLSPClient


@dataclass
class ResolvedName:
    """Internal result of resolving a declaration name to a file location."""

    client: LeanLSPClient
    rel_path: str
    abs_path: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    kind: str
    full_name: str


def _match_name(candidate_name: str, query: str) -> bool:
    """Check if candidate_name matches query (full, short, or qualified name)."""
    if candidate_name == query:
        return True
    # Match on last dotted component (short name) in either direction
    short_candidate = candidate_name.rsplit(".", 1)[-1]
    short_query = query.rsplit(".", 1)[-1]
    if short_candidate == query or short_query == candidate_name:
        return True
    # Suffix match: "Ns.foo" matches "A.Ns.foo" or vice versa
    if candidate_name.endswith("." + query) or query.endswith("." + candidate_name):
        return True
    return False


async def resolve_name(ctx: Context, name: str) -> ResolvedName:
    """Resolve a declaration name to file + range.

    Steps:
      1. lean_local_search(name) → candidates
      2. Filter for exact name match (case-sensitive)
      3. If 0 matches → LeanToolError with suggestion
      4. If >1 matches → prefer project file over .lake/packages, error if still ambiguous
      5. setup_client_for_file(ctx, abs_path) → (client, rel_path)
      6. get_declaration_range(client, rel_path, short_name) → (start_line, end_line)
      7. Return ResolvedName
    """
    name = name.strip()
    if not name:
        raise LeanToolError("Name must not be empty.")

    lifespan = ctx.request_context.lifespan_context
    project_root = lifespan.lean_project_path
    if project_root is None:
        raise LeanToolError(
            "Lean project path not set. Call a file-based tool first, "
            "or pass a file_path to any lean_ tool to initialize the project."
        )

    # Step 1: Search with ripgrep
    # Use the short name (after last dot) as the ripgrep query for better recall
    search_query = name.rsplit(".", 1)[-1]
    raw_results = await asyncio.to_thread(
        lean_local_search,
        query=search_query,
        limit=64,
        project_root=project_root,
    )

    if not raw_results:
        raise LeanToolError(
            f"Declaration '{name}' not found. "
            f"Use lean_local_search to discover available declarations."
        )

    # Step 2: Filter for exact match
    matches = [r for r in raw_results if _match_name(r["name"], name)]

    if not matches:
        # Show top candidates from the search
        top = raw_results[:5]
        candidates = ", ".join(r["name"] for r in top)
        raise LeanToolError(
            f"No exact match for '{name}'. "
            f"Closest candidates: {candidates}. "
            f"Use lean_local_search for broader search."
        )

    # Step 3: Disambiguate
    if len(matches) > 1:
        # Prefer project files over .lake/packages
        project_matches = [
            m for m in matches if not m["file"].startswith(".lake/packages/")
        ]
        if len(project_matches) == 1:
            matches = project_matches
        elif len(project_matches) > 1:
            matches = project_matches
            # If still ambiguous but user gave FQN, try exact FQN match
            fqn_matches = [m for m in matches if m["name"] == name]
            if len(fqn_matches) >= 1:
                matches = fqn_matches[:1]

    if len(matches) > 1:
        candidates = ", ".join(f"{m['name']} ({m['file']})" for m in matches)
        raise LeanToolError(
            f"Ambiguous name '{name}' matches multiple declarations: {candidates}. "
            f"Use the fully qualified name to disambiguate."
        )

    match = matches[0]
    full_name = match["name"]
    kind = match["kind"]
    rel_file = match["file"]

    # Step 5: Setup client
    abs_path = str((project_root / rel_file).resolve())
    rel_path = setup_client_for_file(ctx, abs_path)
    if not rel_path:
        raise LeanToolError(
            f"Unable to start LSP server for file '{rel_file}'."
        )

    client: LeanLSPClient = lifespan.client

    # Step 6: Get declaration range using LSP document symbols
    # Try short name first (namespace blocks), then full name (dotted declarations)
    short_name = full_name.rsplit(".", 1)[-1]
    decl_range = get_declaration_range(client, rel_path, short_name)
    if decl_range is None and short_name != full_name:
        decl_range = get_declaration_range(client, rel_path, full_name)
    if decl_range is None:
        raise LeanToolError(
            f"Declaration '{full_name}' found by search in '{rel_file}' "
            f"but LSP document symbols could not locate it. "
            f"The file may need to be rebuilt (try lean_build)."
        )

    start_line, end_line = decl_range

    return ResolvedName(
        client=client,
        rel_path=rel_path,
        abs_path=abs_path,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        full_name=full_name,
    )
