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

## Local Search (unlimited, no rate limits)
Set env vars for local search without rate limits:
- `LEAN_LEANSEARCH_LOCAL=1`: Semantic search via embeddings (requires chromadb)
- `LEAN_LOOGLE_LOCAL=1`: Type pattern search via local loogle binary

Embedding providers for local leansearch:
- `LEAN_EMBEDDING_PROVIDER=default`: sentence-transformers (no API key, local)
- `LEAN_EMBEDDING_PROVIDER=openai`: OpenAI text-embedding-3-large (best quality)
- `LEAN_EMBEDDING_PROVIDER=gemini`: Google text-embedding-004 (GOOGLE_API_KEY, free tier)
- `LEAN_EMBEDDING_PROVIDER=voyage`: Voyage voyage-code-2 (optimized for code)
- `LEAN_EMBEDDING_MODEL=<model>`: Override default model

## Search Decision Tree
**Default: Use lean_leansearch first** - it handles natural language, type patterns, and concept queries.
Only use specialized tools when leansearch doesn't find what you need.

1. **lean_leansearch** (DEFAULT) - Natural language, concepts, type patterns. Start here!
2. lean_local_search - Verify a specific declaration exists by exact name
3. lean_loogle - Type signature patterns when leansearch misses (e.g., `(_ → _) → List _ → List _`)
4. lean_leanfinder - Conceptual/semantic search as backup
5. lean_state_search - Find lemmas to close current goal
6. lean_hammer_premise - Get premises for simp/aesop

After finding a name: lean_local_search to verify, lean_hover_info for signature.

## Return Formats
List tools return JSON arrays. Empty = `[]`.
"""
