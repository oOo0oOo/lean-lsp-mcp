/-
  LSP Client - Manages connection to Lean language server

  Spawns `lake serve` and communicates via JSON-RPC.
-/
import Lean.Data.Json

namespace LeanMcp.Lsp

open Lean (Json ToJson FromJson JsonNumber)

/-! ## LSP Message Framing -/

/-- Read Content-Length header value -/
private def readContentLength (stream : IO.FS.Stream) : IO (Option Nat) := do
  let mut contentLength : Option Nat := none
  repeat do
    let line ← stream.getLine
    let line := line.trimAscii.toString
    if line.isEmpty then break
    if line.startsWith "Content-Length:" then
      contentLength := line.drop "Content-Length:".length |>.trimAscii.toString |>.toNat?
  return contentLength

/-- Read a single LSP message -/
def readLspMessage (stream : IO.FS.Stream) : IO (Option String) := do
  match ← readContentLength stream with
  | none => return none
  | some len =>
    let bytes ← stream.read len.toUSize
    return some (String.fromUTF8! bytes)

/-- Write an LSP message with Content-Length header -/
def writeLspMessage (stream : IO.FS.Stream) (msg : String) : IO Unit := do
  let bytes := msg.toUTF8
  stream.putStr s!"Content-Length: {bytes.size}\r\n\r\n"
  stream.putStr msg
  stream.flush

/-! ## LSP Types -/

/-- LSP Position (0-indexed) -/
structure Position where
  line : Nat
  character : Nat
  deriving Inhabited, BEq

instance : ToJson Position where
  toJson p := .mkObj [("line", .num p.line), ("character", .num p.character)]

instance : FromJson Position where
  fromJson? j := do
    let line ← j.getObjValAs? Nat "line"
    let character ← j.getObjValAs? Nat "character"
    pure { line, character }

/-- LSP Range -/
structure Range where
  start : Position
  «end» : Position
  deriving Inhabited

instance : ToJson Range where
  toJson r := .mkObj [("start", ToJson.toJson r.start), ("end", ToJson.toJson r.end)]

instance : FromJson Range where
  fromJson? j := do
    let start ← j.getObjValAs? Position "start"
    let end_ ← j.getObjValAs? Position "end"
    pure { start, «end» := end_ }

/-- LSP Diagnostic Severity -/
inductive DiagnosticSeverity
  | error | warning | information | hint
  deriving Inhabited, BEq

instance : FromJson DiagnosticSeverity where
  fromJson? j := do
    let n ← j.getNat?
    match n with
    | 1 => pure .error
    | 2 => pure .warning
    | 3 => pure .information
    | 4 => pure .hint
    | _ => .error "invalid severity"

def DiagnosticSeverity.toString : DiagnosticSeverity → String
  | .error => "error"
  | .warning => "warning"
  | .information => "info"
  | .hint => "hint"

/-- LSP Diagnostic -/
structure Diagnostic where
  range : Range
  severity : Option DiagnosticSeverity := none
  message : String
  deriving Inhabited

instance : FromJson Diagnostic where
  fromJson? j := do
    let range ← j.getObjValAs? Range "range"
    let severity := j.getObjVal? "severity" |>.toOption.bind (FromJson.fromJson? · |>.toOption)
    let message ← j.getObjValAs? String "message"
    pure { range, severity, message }

/-! ## Simple Synchronous LSP Client -/

/-- File URI from path -/
def fileUri (path : System.FilePath) : String :=
  s!"file://{path}"

/-- Simple LSP client state -/
structure ClientState where
  stdin : IO.FS.Stream
  stdout : IO.FS.Stream
  nextId : IO.Ref Nat
  projectPath : System.FilePath

/-- Helper to create JSON number from Nat -/
private def natJson (n : Nat) : Json := .num n

/-- Helper to create position JSON with 0-indexed values -/
private def positionJson (line col : Nat) : Json :=
  .mkObj [("line", natJson (line - 1)), ("character", natJson (col - 1))]

/-- Send request and wait for response -/
partial def ClientState.request (state : ClientState) (method : String) (params : Json) : IO Json := do
  let id ← state.nextId.modifyGet fun n => (n, n + 1)

  let msg := Json.mkObj [
    ("jsonrpc", .str "2.0"),
    ("id", natJson id),
    ("method", .str method),
    ("params", params)
  ]
  writeLspMessage state.stdin msg.compress

  -- Read responses until we get one with our ID
  repeat do
    match ← readLspMessage state.stdout with
    | none => throw (IO.userError "LSP connection closed")
    | some msgStr =>
      match Json.parse msgStr with
      | .error _ => continue
      | .ok json =>
        match json.getObjVal? "id" with
        | .ok idJson =>
          match idJson.getNat? with
          | .ok respId =>
            if respId == id then
              return json.getObjVal? "result" |>.toOption |>.getD .null
          | .error _ => pure ()
        | .error _ => pure ()
  throw (IO.userError "LSP connection closed")

/-- Send notification -/
def ClientState.notify (state : ClientState) (method : String) (params : Json) : IO Unit := do
  let msg := Json.mkObj [
    ("jsonrpc", .str "2.0"),
    ("method", .str method),
    ("params", params)
  ]
  writeLspMessage state.stdin msg.compress

/-! ## Client API -/

/-- Create LSP client for a project -/
def createClient (projectPath : System.FilePath) : IO ClientState := do
  -- Spawn lake serve
  let child ← IO.Process.spawn {
    cmd := "lake"
    args := #["serve"]
    cwd := projectPath
    stdin := .piped
    stdout := .piped
    stderr := .null
  }

  let stdin := IO.FS.Stream.ofHandle child.stdin
  let stdout := IO.FS.Stream.ofHandle child.stdout
  let nextId ← IO.mkRef 0

  let state : ClientState := { stdin, stdout, nextId, projectPath }

  -- Initialize
  let initParams := Json.mkObj [
    ("processId", natJson (← IO.Process.getPID).toNat),
    ("rootUri", .str (fileUri projectPath)),
    ("capabilities", .mkObj [])
  ]
  let _ ← state.request "initialize" initParams
  state.notify "initialized" (.mkObj [])

  return state

/-- Open a document -/
def ClientState.openDocument (state : ClientState) (path : System.FilePath) : IO Unit := do
  let content ← IO.FS.readFile path
  let uri := fileUri path
  let params := Json.mkObj [
    ("textDocument", .mkObj [
      ("uri", .str uri),
      ("languageId", .str "lean4"),
      ("version", natJson 1),
      ("text", .str content)
    ])
  ]
  state.notify "textDocument/didOpen" params

/-- Get hover info at position -/
def ClientState.hover (state : ClientState) (path : System.FilePath) (line col : Nat) : IO (Option String) := do
  let params := Json.mkObj [
    ("textDocument", .mkObj [("uri", .str (fileUri path))]),
    ("position", positionJson line col)
  ]
  let result ← state.request "textDocument/hover" params
  match result.getObjVal? "contents" with
  | .ok contents =>
    match contents.getObjValAs? String "value" with
    | .ok value => return some value
    | .error _ =>
      match contents.getStr? with
      | .ok s => return some s
      | .error _ => return none
  | .error _ => return none

/-- Get completions at position -/
def ClientState.completions (state : ClientState) (path : System.FilePath) (line col : Nat) : IO (Array Json) := do
  let params := Json.mkObj [
    ("textDocument", .mkObj [("uri", .str (fileUri path))]),
    ("position", positionJson line col)
  ]
  let result ← state.request "textDocument/completion" params
  match result.getObjValAs? (Array Json) "items" with
  | .ok items => return items
  | .error _ =>
    match result.getArr? with
    | .ok arr => return arr
    | .error _ => return #[]

/-- Get plain text proof goals at position -/
def ClientState.plainGoal (state : ClientState) (path : System.FilePath) (line col : Nat) : IO (Option (Array String)) := do
  let params := Json.mkObj [
    ("textDocument", .mkObj [("uri", .str (fileUri path))]),
    ("position", positionJson line col)
  ]
  let result ← state.request "$/lean/plainGoal" params

  match result.getObjValAs? (Array String) "goals" with
  | .ok goals => return some goals
  | .error _ =>
    match result.getObjValAs? String "rendered" with
    | .ok rendered => return some #[rendered]
    | .error _ => return none

/-- Get expected type at position -/
def ClientState.plainTermGoal (state : ClientState) (path : System.FilePath) (line col : Nat) : IO (Option String) := do
  let params := Json.mkObj [
    ("textDocument", .mkObj [("uri", .str (fileUri path))]),
    ("position", positionJson line col)
  ]
  let result ← state.request "$/lean/plainTermGoal" params

  match result.getObjValAs? String "goal" with
  | .ok goal => return some goal
  | .error _ => return none

/-- Get definition location -/
def ClientState.definition (state : ClientState) (path : System.FilePath) (line col : Nat) : IO (Option (String × Nat × Nat)) := do
  let params := Json.mkObj [
    ("textDocument", .mkObj [("uri", .str (fileUri path))]),
    ("position", positionJson line col)
  ]
  let result ← state.request "textDocument/definition" params

  let parseLocation (j : Json) : Option (String × Nat × Nat) := do
    let uri ← j.getObjValAs? String "uri" |>.toOption
    let range ← j.getObjVal? "range" |>.toOption
    let start ← range.getObjVal? "start" |>.toOption
    let line ← start.getObjValAs? Nat "line" |>.toOption
    let col ← start.getObjValAs? Nat "character" |>.toOption
    return (uri, line + 1, col + 1)

  match result.getArr? with
  | .ok arr =>
    if h : arr.size > 0 then
      return parseLocation arr[0]
    else
      return none
  | .error _ => return parseLocation result

/-! ## Utilities -/

/-- Get line content from file -/
def getLineContent (path : System.FilePath) (line : Nat) : IO String := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n"
  if line > 0 && line ≤ lines.length then
    return lines[line - 1]!
  else
    return ""

end LeanMcp.Lsp
