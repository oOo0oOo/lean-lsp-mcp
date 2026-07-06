"""Server instructions shown to the agent.

Rate limits are generated from ``config.RATE_LIMITS`` — the same table the
``@rate_limited`` decorators read — so the numbers cannot drift.
"""

from lean_lsp_mcp import config


def _limit(tool: str) -> str:
    max_requests, per_seconds = config.RATE_LIMITS[tool]
    return f"{max_requests}/{per_seconds}s"


INSTRUCTIONS = f"""## General Rules
- All line and column numbers are 1-indexed. Columns count characters (codepoints).
- This MCP does NOT edit files. Use other tools for editing.

## Key Tools
- **lean_goal**: Proof state at position. Omit `column` for before/after. `status` field: 'goals', 'complete' (proof done here), or 'no_goal_at_position' (not inside a proof).
- **lean_diagnostic_messages**: Compiler errors/warnings. "no goals to be solved" = remove tactics.
- **lean_term_goal**: Expected type at a position.
- **lean_hover_info**: Type signature + docs. Column at START of identifier.
- **lean_completions**: IDE autocomplete on incomplete code.
- **lean_local_search**: Fast local declaration search. Use BEFORE trying a lemma name.
- **lean_file_outline**: Token-efficient file skeleton (slow-ish).
- **lean_multi_attempt**: Test tactics without editing at a proof position. Use `column` for an exact source position; omit it for fast line-based attempts: `["simp", "ring", "omega"]`
- **lean_code_actions**: Quick fixes and `TryThis` suggestions (simp?, exact?) with resolved edits.
- **lean_declaration_file**: Declaration source slice with context; `full_file=true` for the whole file (large).
- **lean_references**: All usages of a symbol (capped by `max_results`, `total` reports the full count).
- **lean_run_code**: Run standalone snippet. Must include imports.
- **lean_verify**: Axiom check + source scan. Use fully qualified name (e.g. `Ns.thm`).
- **lean_minimal_hypotheses**: Which explicit hypotheses of a theorem are actually needed.
- **lean_build**: Run `lake build` + restart LSP. Only if needed (new imports). Use `fetch_cache=true` only for missing dependency caches. SLOW!
- **lean_profile_proof**: Profile a theorem for performance. Shows tactic hotspots. SLOW!

## Search Tools (rate limited)
- **lean_leansearch** ({_limit("leansearch")}): Natural language -> mathlib
- **lean_loogle** ({_limit("loogle")}): Type pattern -> mathlib
- **lean_leanfinder** ({_limit("leanfinder")}): Semantic/conceptual search
- **lean_state_search** ({_limit("lean_state_search")}): Goal -> closing lemmas
- **lean_hammer_premise** ({_limit("hammer_premise")}): Goal -> premises for simp/aesop

## Search Decision Tree
1. "Does X exist locally?" -> lean_local_search
2. "I need a lemma that says X" -> lean_leansearch
3. "Find lemma with type pattern" -> lean_loogle
4. "What's the Lean name for concept X?" -> lean_leanfinder
5. "What closes this goal?" -> lean_state_search
6. "What to feed simp?" -> lean_hammer_premise

After finding a name: lean_local_search to verify, lean_hover_info for signature.

## Return Formats
List-returning tools return an object with an `items` array. Empty = `{{"items": []}}`.

## Slow Files
On large files, pass `timeout_s` to lean_diagnostic_messages / lean_goal. A
response with `partial: true` + `still_elaborating_lines` (or goal `status:
'still_elaborating'`) means Lean is still working - poll again; it is NOT an
error or a dead server.

## Error Handling
Check `isError` in responses: `true` means failure (timeout/LSP error/rate limit), while an empty `items` with `isError: false` means no results found.
"""
