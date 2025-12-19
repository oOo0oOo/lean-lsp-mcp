# lean-lsp-mcp Project Instructions

## Quick Status Check After Restart

Run this first to verify everything works:
```bash
uv run scripts/test_mcp_status.py
```

Check status file:
```bash
cat .mcp_status.json
```

## Current Branch & Status

```bash
git branch --show-current  # Should be: feature/repl-integration
git log --oneline -3       # See recent commits
```

## Key Files Modified for Syntax Expansion

- `src/lean_lsp_mcp/syntax_utils.py` - Extracts macro expansion info from InfoTrees
- `src/lean_lsp_mcp/models.py` - MacroExpansion, SyntaxRange models
- `src/lean_lsp_mcp/server.py` - Enhanced hover, goal, diagnostics with expansion info
- `src/lean_lsp_mcp/outline_utils.py` - Detects macro/notation/syntax in outlines

## Test Files

- `tests/test_project/MacroNotation.lean` - Basic macro/notation test
- `tests/test_project/TicTacToe.lean` - DSL stress test (gitignored)

## MCP Configuration

The local MCP is configured in `~/.claude.json`:
```json
{
  "mcpServers": {
    "lean-lsp-mcp": {
      "command": "uvx",
      "args": ["--from", "/Users/alokbeniwal/lean-lsp-mcp", "lean-lsp-mcp"]
    }
  }
}
```

To use the stable published version instead:
```json
{
  "mcpServers": {
    "lean-lsp-default": {
      "command": "uvx",
      "args": ["lean-lsp-mcp"]
    }
  }
}
```

## Running Tests

```bash
# All unit tests
uv run pytest tests/unit/ -v

# Just syntax utils tests
uv run pytest tests/unit/test_syntax_utils.py -v

# Build test Lean project
cd tests/test_project && lake build MacroNotation
```

## Key MCP Tools

After restart, these tools should be available:
- `mcp__lean-lsp-mcp__lean_hover_info` - Get hover info with macro expansion
- `mcp__lean-lsp-mcp__lean_file_outline` - File outline with syntax tags
- `mcp__lean-lsp-mcp__lean_goal` - Proof goals with tactic expansion
- `mcp__lean-lsp-mcp__lean_diagnostic_messages` - Diagnostics with macro context

## Debugging MCP Connection

If MCP tools aren't available after restart:
1. Check Claude Code logs for MCP connection errors
2. Try running the server manually: `uv run lean-lsp-mcp`
3. Verify config: `cat ~/.claude.json | jq '.mcpServers["lean-lsp-mcp"]'`

## Current Work: Syntax Expansion Feature

The goal is to help AI agents understand Lean 4 custom syntax automatically.

**Completed:**
- Created syntax_utils.py for InfoTree parsing
- Enhanced hover_info with macro expansion
- Enhanced file_outline with is_macro, is_notation flags
- Enhanced goal with tactic_expansion
- Enhanced diagnostic_messages with macro context
- Created TicTacToe DSL stress test

**InfoTree Line Coordinates:**
- InfoTree line = 1-indexed file line + 1
- InfoTree column = 0-indexed
- To query: `get_macro_expansion_at_position(trees, line + 1, column - 1)`
