/-
  Core LSP-based tools

  These tools interact with the Lean language server to provide:
  - Proof goals
  - Diagnostics
  - Hover information
  - Completions
  - Definition locations
-/
import LeanMcp.Mcp.Protocol
import LeanMcp.Lsp

namespace LeanMcp.Tools

open LeanMcp.Mcp (ToolDef ToolInputSchema ToolResult ContentType)
open LeanMcp.Lsp (ClientState)
open Lean (Json)

/-! ## Tool Context -/

/-- Shared context for all tools -/
structure ToolContext where
  client : IO.Ref (Option ClientState)
  projectPath : IO.Ref (Option System.FilePath)

/-- Initialize tool context -/
def ToolContext.new : IO ToolContext := do
  let client ← IO.mkRef none
  let projectPath ← IO.mkRef none
  return { client, projectPath }

/-- Get or create LSP client for a file -/
def ToolContext.getClient (ctx : ToolContext) (filePath : System.FilePath) : IO ClientState := do
  -- Find project root by looking for lean-toolchain
  let projectPath ← findProjectRoot filePath
  let currentProject ← ctx.projectPath.get

  -- If same project, reuse client
  if currentProject == some projectPath then
    if let some client := ← ctx.client.get then
      return client

  -- Create new client
  let client ← LeanMcp.Lsp.createClient projectPath
  ctx.client.set (some client)
  ctx.projectPath.set (some projectPath)
  return client
where
  findProjectRoot (path : System.FilePath) : IO System.FilePath := do
    let mut current := if ← path.isDir then path else path.parent.getD "."
    while current != "/" do
      if ← (current / "lean-toolchain").pathExists then
        return current
      if ← (current / "lakefile.lean").pathExists then
        return current
      if ← (current / "lakefile.toml").pathExists then
        return current
      current := current.parent.getD "/"
    throw (IO.userError s!"Could not find Lean project root for {path}")

/-! ## Tool Result Helpers -/

def errorResult (s : String) : ToolResult :=
  { content := #[.text s], isError := true }

def jsonResult (j : Json) : ToolResult :=
  { content := #[.text j.compress], isError := false }

/-! ## lean_goal -/

def goalToolDef : ToolDef := {
  name := "lean_goal"
  description := "Get proof goals at a position. MOST IMPORTANT tool - use often!\n\nOmit column to see goals_before (line start) and goals_after (line end), showing how the tactic transforms the state. \"no goals\" = proof complete."
  inputSchema := {
    properties := .mkObj [
      ("file_path", .mkObj [("type", .str "string"), ("description", .str "Absolute path to Lean file")]),
      ("line", .mkObj [("type", .str "integer"), ("description", .str "Line number (1-indexed)")]),
      ("column", .mkObj [("type", .str "integer"), ("description", .str "Column (1-indexed). Omit for before/after")])
    ]
    required := #["file_path", "line"]
  }
}

def goalHandler (ctx : ToolContext) (params : Json) : IO ToolResult := do
  let some filePath := params.getObjValAs? String "file_path" |>.toOption
    | return errorResult "Missing file_path"
  let some line := params.getObjValAs? Nat "line" |>.toOption
    | return errorResult "Missing line"
  let column := params.getObjValAs? Nat "column" |>.toOption

  let client ← ctx.getClient filePath

  -- Open document if not already open
  client.openDocument filePath

  -- Get line content for context
  let lineContent ← LeanMcp.Lsp.getLineContent filePath line

  match column with
  | some col =>
    -- Single position query
    let goals ← client.plainGoal filePath line col
    let goalsJson := match goals with
      | some g => Json.arr (g.map Json.str)
      | none => .null
    let result := Json.mkObj [
      ("line_context", .str lineContent),
      ("goals", goalsJson)
    ]
    return jsonResult result

  | none =>
    -- Before/after query (line start and end)
    let goalsBefore ← client.plainGoal filePath line 1
    let goalsAfter ← client.plainGoal filePath line (lineContent.length + 1)
    let goalsBeforeJson := match goalsBefore with
      | some g => Json.arr (g.map Json.str)
      | none => .null
    let goalsAfterJson := match goalsAfter with
      | some g => Json.arr (g.map Json.str)
      | none => .null
    let result := Json.mkObj [
      ("line_context", .str lineContent),
      ("goals_before", goalsBeforeJson),
      ("goals_after", goalsAfterJson)
    ]
    return jsonResult result

/-! ## lean_diagnostic_messages -/

def diagnosticsToolDef : ToolDef := {
  name := "lean_diagnostic_messages"
  description := "Get compiler diagnostics (errors, warnings, infos) for a Lean file."
  inputSchema := {
    properties := .mkObj [
      ("file_path", .mkObj [("type", .str "string"), ("description", .str "Absolute path to Lean file")]),
      ("start_line", .mkObj [("type", .str "integer"), ("description", .str "Filter from line (1-indexed)")]),
      ("end_line", .mkObj [("type", .str "integer"), ("description", .str "Filter to line (1-indexed)")])
    ]
    required := #["file_path"]
  }
}

def diagnosticsHandler (ctx : ToolContext) (params : Json) : IO ToolResult := do
  let some filePath := params.getObjValAs? String "file_path" |>.toOption
    | return errorResult "Missing file_path"
  let _startLine := params.getObjValAs? Nat "start_line" |>.toOption
  let _endLine := params.getObjValAs? Nat "end_line" |>.toOption

  let client ← ctx.getClient filePath
  client.openDocument filePath

  -- Wait a bit for diagnostics to arrive
  IO.sleep 500

  -- For now, return empty diagnostics (would need LSP notification handling for real diagnostics)
  return jsonResult (.mkObj [("items", .arr #[])])

/-! ## lean_hover_info -/

def hoverToolDef : ToolDef := {
  name := "lean_hover_info"
  description := "Get type signature and docs for a symbol. Essential for understanding APIs."
  inputSchema := {
    properties := .mkObj [
      ("file_path", .mkObj [("type", .str "string"), ("description", .str "Absolute path to Lean file")]),
      ("line", .mkObj [("type", .str "integer"), ("description", .str "Line number (1-indexed)")]),
      ("column", .mkObj [("type", .str "integer"), ("description", .str "Column at START of identifier")])
    ]
    required := #["file_path", "line", "column"]
  }
}

def hoverHandler (ctx : ToolContext) (params : Json) : IO ToolResult := do
  let some filePath := params.getObjValAs? String "file_path" |>.toOption
    | return errorResult "Missing file_path"
  let some line := params.getObjValAs? Nat "line" |>.toOption
    | return errorResult "Missing line"
  let some column := params.getObjValAs? Nat "column" |>.toOption
    | return errorResult "Missing column"

  let client ← ctx.getClient filePath
  client.openDocument filePath

  let hover ← client.hover filePath line column

  let result := Json.mkObj [
    ("info", match hover with | some h => .str h | none => .null),
    ("diagnostics", .arr #[])
  ]
  return jsonResult result

/-! ## lean_completions -/

def completionsToolDef : ToolDef := {
  name := "lean_completions"
  description := "Get IDE autocompletions. Use on INCOMPLETE code (after `.` or partial name)."
  inputSchema := {
    properties := .mkObj [
      ("file_path", .mkObj [("type", .str "string"), ("description", .str "Absolute path to Lean file")]),
      ("line", .mkObj [("type", .str "integer"), ("description", .str "Line number (1-indexed)")]),
      ("column", .mkObj [("type", .str "integer"), ("description", .str "Column number (1-indexed)")]),
      ("max_completions", .mkObj [("type", .str "integer"), ("description", .str "Max completions (default 32)")])
    ]
    required := #["file_path", "line", "column"]
  }
}

def completionsHandler (ctx : ToolContext) (params : Json) : IO ToolResult := do
  let some filePath := params.getObjValAs? String "file_path" |>.toOption
    | return errorResult "Missing file_path"
  let some line := params.getObjValAs? Nat "line" |>.toOption
    | return errorResult "Missing line"
  let some column := params.getObjValAs? Nat "column" |>.toOption
    | return errorResult "Missing column"
  let maxCompletions := params.getObjValAs? Nat "max_completions" |>.toOption |>.getD 32

  let client ← ctx.getClient filePath
  client.openDocument filePath

  let completions ← client.completions filePath line column

  let items := (completions.toSubarray 0 (min maxCompletions completions.size)).toArray.map fun c =>
    let label := c.getObjValAs? String "label" |>.toOption |>.getD ""
    let kind := c.getObjValAs? Nat "kind" |>.toOption |>.map completionKindName
    let detail := c.getObjValAs? String "detail" |>.toOption
    Json.mkObj ([
      ("label", .str label)
    ] ++ (kind.map (("kind", .str ·)) |>.toList) ++ (detail.map (("detail", .str ·)) |>.toList))

  return jsonResult (.mkObj [("items", .arr items)])
where
  completionKindName : Nat → String
    | 1 => "Text" | 2 => "Method" | 3 => "Function" | 4 => "Constructor"
    | 5 => "Field" | 6 => "Variable" | 7 => "Class" | 8 => "Interface"
    | 9 => "Module" | 10 => "Property" | 11 => "Unit" | 12 => "Value"
    | 13 => "Enum" | 14 => "Keyword" | 15 => "Snippet" | 16 => "Color"
    | 17 => "File" | 18 => "Reference" | 19 => "Folder" | 20 => "EnumMember"
    | 21 => "Constant" | 22 => "Struct" | 23 => "Event" | 24 => "Operator"
    | 25 => "TypeParameter" | _ => "Unknown"

/-! ## lean_term_goal -/

def termGoalToolDef : ToolDef := {
  name := "lean_term_goal"
  description := "Get the expected type at a position."
  inputSchema := {
    properties := .mkObj [
      ("file_path", .mkObj [("type", .str "string"), ("description", .str "Absolute path to Lean file")]),
      ("line", .mkObj [("type", .str "integer"), ("description", .str "Line number (1-indexed)")]),
      ("column", .mkObj [("type", .str "integer"), ("description", .str "Column (defaults to end of line)")])
    ]
    required := #["file_path", "line"]
  }
}

def termGoalHandler (ctx : ToolContext) (params : Json) : IO ToolResult := do
  let some filePath := params.getObjValAs? String "file_path" |>.toOption
    | return errorResult "Missing file_path"
  let some line := params.getObjValAs? Nat "line" |>.toOption
    | return errorResult "Missing line"

  let client ← ctx.getClient filePath
  client.openDocument filePath

  let lineContent ← LeanMcp.Lsp.getLineContent filePath line
  let column := params.getObjValAs? Nat "column" |>.toOption |>.getD (lineContent.length + 1)

  let goal ← client.plainTermGoal filePath line column

  let result := Json.mkObj [
    ("goal", match goal with | some g => .str g | none => .null)
  ]
  return jsonResult result

/-! ## lean_declaration_file -/

def declarationFileToolDef : ToolDef := {
  name := "lean_declaration_file"
  description := "Get file where a symbol is declared. Symbol must be present in file first."
  inputSchema := {
    properties := .mkObj [
      ("file_path", .mkObj [("type", .str "string"), ("description", .str "Absolute path to Lean file")]),
      ("symbol", .mkObj [("type", .str "string"), ("description", .str "Symbol (case sensitive, must be in file)")])
    ]
    required := #["file_path", "symbol"]
  }
}

def declarationFileHandler (ctx : ToolContext) (params : Json) : IO ToolResult := do
  let some filePath := params.getObjValAs? String "file_path" |>.toOption
    | return errorResult "Missing file_path"
  let some symbol := params.getObjValAs? String "symbol" |>.toOption
    | return errorResult "Missing symbol"

  let client ← ctx.getClient filePath
  client.openDocument filePath

  -- Find the symbol in the file
  let content ← IO.FS.readFile filePath
  let lines := content.splitOn "\n"

  -- Search for the symbol
  let mut lineNum : Nat := 0
  for lineContent in lines do
    lineNum := lineNum + 1
    -- Check if symbol exists in line and find its position
    let parts := lineContent.splitOn symbol
    if parts.length > 1 then
      -- Symbol found, position is length of first part + 1 (1-indexed)
      let colIdx := (parts.head?.getD "").length + 1
      let location ← client.definition filePath lineNum colIdx
      match location with
      | some (uri, ln, col) =>
        -- Convert file URI to path
        let path := if uri.startsWith "file://" then (uri.drop 7).toString else uri
        return jsonResult (.mkObj [
          ("file", .str path),
          ("line", .num ln),
          ("column", .num col)
        ])
      | none => continue

  return errorResult s!"Symbol '{symbol}' not found in file"

end LeanMcp.Tools
