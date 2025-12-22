# lean-lsp-mcp Project Instructions

## Quick Status Check

```bash
uv run scripts/test_mcp_status.py
```

## Key Files for Syntax Expansion

- `src/lean_lsp_mcp/syntax_utils.py` - Extracts macro expansion info from InfoTrees
- `src/lean_lsp_mcp/models.py` - MacroExpansion, SyntaxRange models
- `src/lean_lsp_mcp/server.py` - Enhanced hover, goal, diagnostics with expansion info
- `src/lean_lsp_mcp/outline_utils.py` - Detects macro/notation/syntax in outlines

## Running Tests

```bash
uv run pytest tests/unit/ -v
uv run pytest tests/unit/test_syntax_utils.py -v
cd tests/test_project && lake build MacroNotation
```

## InfoTree Coordinates

InfoTree line numbers have a variable offset from file lines. Use text-based matching:
- `get_macro_expansion_by_text(trees, source_text)` - reliable
- Position-based lookup is unreliable due to offset drift
