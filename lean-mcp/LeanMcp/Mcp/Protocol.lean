/-
  MCP (Model Context Protocol) JSON-RPC implementation

  MCP uses JSON-RPC 2.0 over stdio with Content-Length headers (like LSP).
-/
import Lean.Data.Json

namespace LeanMcp.Mcp

open Lean (Json ToJson FromJson JsonNumber)

/-! ## JSON-RPC Types -/

/-- JSON-RPC request ID (can be string or number) -/
inductive RequestId
  | str (s : String)
  | num (n : Int)
  deriving Inhabited, BEq

instance : ToJson RequestId where
  toJson
    | .str s => .str s
    | .num n => .num (JsonNumber.fromInt n)

instance : FromJson RequestId where
  fromJson? j := match j with
    | .str s => pure (.str s)
    | .num n => pure (.num n.mantissa)
    | _ => .error "expected string or number for request id"

/-- JSON-RPC request message -/
structure Request where
  jsonrpc : String := "2.0"
  id : Option RequestId := none
  method : String
  params : Option Json := none
  deriving Inhabited

instance : FromJson Request where
  fromJson? j := do
    let jsonrpc ← j.getObjValAs? String "jsonrpc"
    let id := j.getObjVal? "id" |>.bind FromJson.fromJson? |>.toOption
    let method ← j.getObjValAs? String "method"
    let params := j.getObjVal? "params" |>.toOption
    pure { jsonrpc, id, method, params }

/-- JSON-RPC success response -/
structure Response where
  jsonrpc : String := "2.0"
  id : RequestId
  result : Json
  deriving Inhabited

instance : ToJson Response where
  toJson r := .mkObj [
    ("jsonrpc", .str r.jsonrpc),
    ("id", ToJson.toJson r.id),
    ("result", r.result)
  ]

/-- JSON-RPC error object -/
structure ErrorData where
  code : Int
  message : String
  data : Option Json := none
  deriving Inhabited

instance : ToJson ErrorData where
  toJson e :=
    let base := [("code", .num (JsonNumber.fromInt e.code)), ("message", .str e.message)]
    let fields := match e.data with
      | some d => base ++ [("data", d)]
      | none => base
    .mkObj fields

/-- JSON-RPC error response -/
structure ErrorResponse where
  jsonrpc : String := "2.0"
  id : Option RequestId
  error : ErrorData
  deriving Inhabited

instance : ToJson ErrorResponse where
  toJson r :=
    let idJson := match r.id with
      | some id => ToJson.toJson id
      | none => .null
    .mkObj [
      ("jsonrpc", .str r.jsonrpc),
      ("id", idJson),
      ("error", ToJson.toJson r.error)
    ]

/-! ## Standard JSON-RPC Error Codes -/

def ErrorCode.parseError : Int := -32700
def ErrorCode.invalidRequest : Int := -32600
def ErrorCode.methodNotFound : Int := -32601
def ErrorCode.invalidParams : Int := -32602
def ErrorCode.internalError : Int := -32603

/-! ## MCP-Specific Types -/

/-- MCP protocol version -/
def protocolVersion : String := "2024-11-05"

/-- Server capabilities -/
structure ServerCapabilities where
  tools : Option Json := some (.mkObj [("listChanged", .bool true)])
  resources : Option Json := some (.mkObj [("subscribe", .bool false), ("listChanged", .bool true)])
  prompts : Option Json := none
  logging : Option Json := none
  deriving Inhabited

instance : ToJson ServerCapabilities where
  toJson c :=
    let fields := #[
      c.tools.map (("tools", ·)),
      c.resources.map (("resources", ·)),
      c.prompts.map (("prompts", ·)),
      c.logging.map (("logging", ·))
    ].filterMap id
    .mkObj fields.toList

/-- Server info -/
structure ServerInfo where
  name : String := "lean-mcp"
  version : String := "0.1.0"
  deriving Inhabited

instance : ToJson ServerInfo where
  toJson s := .mkObj [
    ("name", .str s.name),
    ("version", .str s.version)
  ]

/-- Initialize result -/
structure InitializeResult where
  protocolVersion : String := Mcp.protocolVersion
  capabilities : ServerCapabilities := {}
  serverInfo : ServerInfo := {}
  deriving Inhabited

instance : ToJson InitializeResult where
  toJson r := .mkObj [
    ("protocolVersion", .str r.protocolVersion),
    ("capabilities", ToJson.toJson r.capabilities),
    ("serverInfo", ToJson.toJson r.serverInfo)
  ]

/-! ## Tool Types -/

/-- Tool parameter schema (JSON Schema) -/
structure ToolInputSchema where
  type : String := "object"
  properties : Json := .mkObj []
  required : Array String := #[]
  deriving Inhabited

instance : ToJson ToolInputSchema where
  toJson s := .mkObj [
    ("type", .str s.type),
    ("properties", s.properties),
    ("required", .arr (s.required.map .str))
  ]

/-- Tool definition -/
structure ToolDef where
  name : String
  description : String
  inputSchema : ToolInputSchema
  deriving Inhabited

instance : ToJson ToolDef where
  toJson t := .mkObj [
    ("name", .str t.name),
    ("description", .str t.description),
    ("inputSchema", ToJson.toJson t.inputSchema)
  ]

/-- Content types for tool results -/
inductive ContentType
  | text (text : String)
  | image (data : String) (mimeType : String)
  | resource (uri : String) (mimeType : Option String) (text : Option String)
  deriving Inhabited

instance : ToJson ContentType where
  toJson
    | .text t => .mkObj [("type", .str "text"), ("text", .str t)]
    | .image d m => .mkObj [("type", .str "image"), ("data", .str d), ("mimeType", .str m)]
    | .resource u m t =>
      let base := [("type", .str "resource"), ("uri", .str u)]
      let fields := base ++
        (m.map (("mimeType", .str ·)) |>.toList) ++
        (t.map (("text", .str ·)) |>.toList)
      .mkObj fields

/-- Tool call result -/
structure ToolResult where
  content : Array ContentType
  isError : Bool := false
  deriving Inhabited

instance : ToJson ToolResult where
  toJson r := .mkObj [
    ("content", .arr (r.content.map ToJson.toJson)),
    ("isError", .bool r.isError)
  ]

/-! ## Resource Types -/

/-- Resource definition -/
structure ResourceDef where
  uri : String
  name : String
  description : Option String := none
  mimeType : String := "text/plain"
  deriving Inhabited

instance : ToJson ResourceDef where
  toJson r :=
    let base := [
      ("uri", .str r.uri),
      ("name", .str r.name),
      ("mimeType", .str r.mimeType)
    ]
    let fields := match r.description with
      | some d => base ++ [("description", .str d)]
      | none => base
    .mkObj fields

/-- Resource content -/
structure ResourceContent where
  uri : String
  mimeType : String := "text/plain"
  text : Option String := none
  blob : Option String := none
  deriving Inhabited

instance : ToJson ResourceContent where
  toJson r :=
    let base := [("uri", .str r.uri), ("mimeType", .str r.mimeType)]
    let fields := base ++
      (r.text.map (("text", .str ·)) |>.toList) ++
      (r.blob.map (("blob", .str ·)) |>.toList)
    .mkObj fields

end LeanMcp.Mcp
