/-
  MCP Server implementation

  Handles stdio communication with Content-Length headers (LSP-style framing).
-/
import LeanMcp.Mcp.Protocol

namespace LeanMcp.Mcp

open Lean (Json ToJson FromJson)

/-! ## Message Framing -/

/-- Read a single message from stdin (Content-Length framing) -/
partial def readMessage (stdin : IO.FS.Stream) : IO (Option String) := do
  -- Read headers until empty line
  let mut contentLength : Option Nat := none
  repeat do
    let line ← stdin.getLine
    let line := line.trimAscii.toString
    if line.isEmpty then
      break
    if line.startsWith "Content-Length:" then
      let lenStr := line.drop "Content-Length:".length |>.trimAscii.toString
      contentLength := lenStr.toNat?

  match contentLength with
  | none => return none
  | some len =>
    -- Read exactly `len` bytes
    let bytes ← stdin.read len.toUSize
    return some (String.fromUTF8! bytes)

/-- Write a message to stdout with Content-Length header -/
def writeMessage (stdout : IO.FS.Stream) (msg : String) : IO Unit := do
  let bytes := msg.toUTF8
  stdout.putStr s!"Content-Length: {bytes.size}\r\n\r\n"
  stdout.putStr msg
  stdout.flush

/-- Send a JSON-RPC response -/
def sendResponse (stdout : IO.FS.Stream) (resp : Response) : IO Unit := do
  writeMessage stdout (ToJson.toJson resp).compress

/-- Send a JSON-RPC error response -/
def sendError (stdout : IO.FS.Stream) (id : Option RequestId) (code : Int) (message : String) : IO Unit := do
  let resp : ErrorResponse := {
    id := id
    error := { code, message }
  }
  writeMessage stdout (ToJson.toJson resp).compress

/-! ## Server State -/

/-- Tool handler function type -/
def ToolHandler := Json → IO ToolResult

/-- Server configuration -/
structure ServerConfig where
  name : String := "lean-mcp"
  version : String := "0.1.0"
  projectPath : Option System.FilePath := none
  deriving Inhabited

/-- Mutable server state -/
structure ServerState where
  initialized : Bool := false
  tools : Std.HashMap String (ToolDef × ToolHandler) := {}
  resources : Array ResourceDef := #[]

abbrev ServerM := StateT ServerState IO

/-! ## Request Handlers -/

/-- Handle initialize request -/
def handleInitialize (_params : Option Json) : ServerM Json := do
  modify fun s => { s with initialized := true }
  let result : InitializeResult := {}
  return ToJson.toJson result

/-- Handle tools/list request -/
def handleToolsList (_params : Option Json) : ServerM Json := do
  let state ← get
  let tools := state.tools.fold (init := #[]) fun acc _ (def_, _) => acc.push (ToJson.toJson def_)
  return .mkObj [("tools", .arr tools)]

/-- Handle tools/call request -/
def handleToolsCall (params : Option Json) : ServerM Json := do
  let state ← get
  let some params := params | return ToJson.toJson ({ content := #[.text "Missing params"], isError := true } : ToolResult)

  let some name := params.getObjVal? "name" |>.toOption.bind (·.getStr?.toOption)
    | return ToJson.toJson ({ content := #[.text "Missing tool name"], isError := true } : ToolResult)

  let args := params.getObjVal? "arguments" |>.toOption |>.getD (.mkObj [])

  match state.tools.get? name with
  | none =>
    return ToJson.toJson ({ content := #[.text s!"Unknown tool: {name}"], isError := true } : ToolResult)
  | some (_, handler) =>
    let result ← liftM (handler args)
    return ToJson.toJson result

/-- Handle resources/list request -/
def handleResourcesList (_params : Option Json) : ServerM Json := do
  let state ← get
  let resources := state.resources.map ToJson.toJson
  return .mkObj [("resources", .arr resources)]

/-- Handle notifications (no response needed) -/
def handleNotification (_method : String) (_params : Option Json) : ServerM Unit := do
  -- Notifications like "initialized" don't need a response
  pure ()

/-! ## Server Main Loop -/

/-- Route a request to the appropriate handler -/
def routeRequest (method : String) (params : Option Json) : ServerM Json := do
  match method with
  | "initialize" => handleInitialize params
  | "tools/list" => handleToolsList params
  | "tools/call" => handleToolsCall params
  | "resources/list" => handleResourcesList params
  | "resources/read" =>
    -- TODO: implement resource reading
    return .mkObj [("contents", .arr #[])]
  | "ping" =>
    return .mkObj []
  | _ =>
    throw <| IO.userError s!"Unknown method: {method}"

/-- Process a single message -/
def processMessage (stdout : IO.FS.Stream) (msgStr : String) : ServerM Unit := do
  match Json.parse msgStr with
  | .error e =>
    sendError stdout none ErrorCode.parseError s!"Parse error: {e}"
  | .ok json =>
    match (FromJson.fromJson? json : Except String Request) with
    | .error e =>
      sendError stdout none ErrorCode.invalidRequest s!"Invalid request: {e}"
    | .ok req =>
      -- Check if this is a notification (no id)
      match req.id with
      | none =>
        handleNotification req.method req.params
      | some id =>
        try
          let result ← routeRequest req.method req.params
          sendResponse stdout { id, result }
        catch e =>
          sendError stdout (some id) ErrorCode.internalError e.toString

/-- Register a tool with the server -/
def registerTool (def_ : ToolDef) (handler : ToolHandler) : ServerM Unit := do
  modify fun s => { s with tools := s.tools.insert def_.name (def_, handler) }

/-- Main server loop -/
partial def runServer (config : ServerConfig) (setup : ServerM Unit) : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout

  let initialState : ServerState := {}
  let ((), state) ← setup.run initialState

  let rec loop (state : ServerState) : IO Unit := do
    match ← readMessage stdin with
    | none => return () -- EOF
    | some msg =>
      let ((), state') ← (processMessage stdout msg).run state
      loop state'

  loop state

end LeanMcp.Mcp
