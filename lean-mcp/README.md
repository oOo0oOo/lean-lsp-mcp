# lean-mcp

Pure Lean 4 implementation of the MCP server for Lean language server integration.

## Building

```bash
lake build
```

## Usage

```bash
# Run server (for use with MCP clients)
.lake/build/bin/leanmcp

# With verbose output
.lake/build/bin/leanmcp --verbose

# Specify project root
.lake/build/bin/leanmcp --project /path/to/lean/project
```

## Tools

### Core LSP Tools
- **lean_goal**: Get proof goals at a position
- **lean_diagnostic_messages**: Get compiler diagnostics
- **lean_hover_info**: Get type signature and docs
- **lean_completions**: Get IDE autocompletions
- **lean_term_goal**: Get expected type at position
- **lean_declaration_file**: Get source file for a symbol

### Search Tools (rate limited)
- **lean_leansearch** (3/30s): Natural language search via leansearch.net
- **lean_loogle** (3/30s): Type signature search via loogle.lean-lang.org
- **lean_leanfinder** (10/30s): Semantic search via LeanFinder
- **lean_local_search**: Fast local ripgrep search

## Architecture

```
lean-mcp/
├── Main.lean              # CLI entry point
├── LeanMcp.lean           # Root module
├── LeanMcp/
│   ├── Json.lean          # JSON utilities
│   ├── Mcp/
│   │   ├── Protocol.lean  # MCP types (Request, Response, ToolDef)
│   │   └── Server.lean    # JSON-RPC server (stdio transport)
│   ├── Lsp/
│   │   └── Client.lean    # LSP client (spawns lake serve)
│   ├── Tools/
│   │   └── Core.lean      # Core LSP-based tools
│   └── Search/
│       └── External.lean  # External API integrations
└── lakefile.lean          # Lake build config
```
