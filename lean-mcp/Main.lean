/-
  lean-mcp: Native Lean 4 MCP Server

  A pure Lean 4 implementation of the MCP server for Lean language server integration.

  Usage:
    leanmcp                    # Run server with stdio transport
    leanmcp --project /path    # Specify Lean project path
    leanmcp --help             # Show help
-/
import Cli
import LeanMcp

open Cli
open LeanMcp.Mcp
open LeanMcp.Tools
open LeanMcp.Search

/-- Register all core LSP tools -/
def registerCoreTools (toolCtx : ToolContext) : ServerM Unit := do
  -- Goal
  registerTool goalToolDef (goalHandler toolCtx)

  -- Diagnostics
  registerTool diagnosticsToolDef (diagnosticsHandler toolCtx)

  -- Hover
  registerTool hoverToolDef (hoverHandler toolCtx)

  -- Completions
  registerTool completionsToolDef (completionsHandler toolCtx)

  -- Term goal
  registerTool termGoalToolDef (termGoalHandler toolCtx)

  -- Declaration file
  registerTool declarationFileToolDef (declarationFileHandler toolCtx)

/-- Register all search tools -/
def registerSearchTools (searchCtx : SearchContext) : ServerM Unit := do
  -- LeanSearch
  registerTool leansearchToolDef (leansearchHandler searchCtx.leansearchLimiter)

  -- Loogle
  registerTool loogleToolDef (loogleHandler searchCtx.loogleLimiter)

  -- LeanFinder
  registerTool leanfinderToolDef (leanfinderHandler searchCtx.leanfinderLimiter)

  -- Local search
  registerTool localSearchToolDef localSearchHandler

/-- Server instructions for MCP clients -/
def serverInstructions : String :=
"## General Rules
- All line and column numbers are 1-indexed.
- This MCP does NOT edit files. Use other tools for editing.

## Key Tools
- **lean_goal**: Proof state at position. Omit `column` for before/after. \"no goals\" = done!
- **lean_diagnostic_messages**: Compiler errors/warnings. \"no goals to be solved\" = remove tactics.
- **lean_hover_info**: Type signature + docs. Column at START of identifier.
- **lean_completions**: IDE autocomplete on incomplete code.
- **lean_local_search**: Fast local declaration search. Use BEFORE trying a lemma name.

## Search Tools (rate limited)
- **lean_leansearch** (3/30s): Natural language → mathlib
- **lean_loogle** (3/30s): Type pattern → mathlib
- **lean_leanfinder** (10/30s): Semantic/conceptual search

## Search Decision Tree
1. \"Does X exist locally?\" → lean_local_search
2. \"I need a lemma that says X\" → lean_leansearch
3. \"Find lemma with type pattern\" → lean_loogle
4. \"What's the Lean name for concept X?\" → lean_leanfinder

After finding a name: lean_local_search to verify, lean_hover_info for signature."

/-- Run the MCP server -/
def runLeanMcp (args : Parsed) : IO UInt32 := do
  let projectPath := args.flag? "project" |>.map (·.as! String)
  let verbose := args.hasFlag "verbose"

  if verbose then
    IO.eprintln "Starting lean-mcp server..."
    IO.eprintln s!"Project path: {projectPath.getD "<auto-detect>"}"

  -- Initialize contexts
  let toolCtx ← ToolContext.new
  let searchCtx ← SearchContext.new

  -- Set initial project path if provided
  if let some path := projectPath then
    toolCtx.projectPath.set (some path)

  -- Configure and run server
  let config : ServerConfig := {
    name := "lean-mcp"
    version := "0.1.0"
    projectPath := projectPath.map (⟨·⟩)
  }

  let setup : ServerM Unit := do
    registerCoreTools toolCtx
    registerSearchTools searchCtx

  runServer config setup
  return 0

/-- Main CLI command -/
def leanMcpCmd : Cmd := `[Cli|
  leanmcp VIA runLeanMcp; ["0.1.0"]
  "Native Lean 4 MCP server for language server integration."

  FLAGS:
    p, project : String; "Path to Lean project root"
    v, verbose;          "Enable verbose logging"

  EXTENSIONS:
    author "lean-lsp-mcp"
]

/-- Entry point -/
def main (args : List String) : IO UInt32 := do
  leanMcpCmd.validate args
