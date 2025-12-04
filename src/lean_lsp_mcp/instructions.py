INSTRUCTIONS = """## General Rules
- All line and column numbers are 1-indexed.
- This MCP does NOT edit files. Use other tools for editing.

## Key Tools
- **lean_goal**: Proof state at position. Omit `column` for before/after. "no goals" = done!
- **lean_diagnostic_messages**: Compiler errors/warnings. "no goals to be solved" = remove tactics.
- **lean_hover_info**: Type signature + docs. Column at START of identifier.
- **lean_completions**: IDE autocomplete on incomplete code.
- **lean_local_search**: Fast local declaration search. Use BEFORE trying a lemma name.
- **lean_file_outline**: Token-efficient file skeleton (slow-ish).
- **lean_multi_attempt**: Test tactics without editing: `["simp", "ring", "omega"]`
- **lean_declaration_file**: Get declaration source. Use sparingly (large output).
- **lean_run_code**: Run standalone snippet. Use rarely.
- **lean_build**: Rebuild + restart LSP. Only if needed (new imports). SLOW!

## Search Tools (rate limited)
- **lean_leansearch** (3/30s): Natural language → mathlib
- **lean_loogle** (3/30s): Type pattern → mathlib
- **lean_leanfinder** (10/30s): Semantic/conceptual search
- **lean_state_search** (3/30s): Goal → closing lemmas
- **lean_hammer_premise** (3/30s): Goal → premises for simp/aesop

## Search Decision Tree
1. "Does X exist locally?" → lean_local_search
2. "I need a lemma that says X" → lean_leansearch
3. "Find lemma with type pattern" → lean_loogle
4. "What's the Lean name for concept X?" → lean_leanfinder
5. "What closes this goal?" → lean_state_search
6. "What to feed simp?" → lean_hammer_premise

After finding a name: lean_local_search to verify, lean_hover_info for signature.

## Return Formats
List tools return JSON arrays. Empty = `[]`.
"""
