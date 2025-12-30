/-
  JSON utilities for MCP protocol

  Most JSON operations use Lean.Data.Json directly.
  This module provides convenience re-exports.
-/
import Lean.Data.Json

namespace LeanMcp.Json

export Lean (Json ToJson FromJson)
export Lean.Json (parse compress)

end LeanMcp.Json
