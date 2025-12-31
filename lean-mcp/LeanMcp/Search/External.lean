/-
  External search API integrations

  Uses curl for HTTP requests to:
  - LeanSearch (leansearch.net)
  - Loogle (loogle.lean-lang.org)
  - LeanFinder (HuggingFace)
  - State Search (premise-search.com)
  - Hammer Premise (leanpremise.net)
-/
import Lean.Data.Json
import LeanMcp.Mcp.Protocol

namespace LeanMcp.Search

open Lean (Json)
open LeanMcp.Mcp (ToolDef ToolInputSchema ToolResult ContentType)

/-! ## URL Encoding -/

/-- Percent-encode a character if needed for URLs -/
private def encodeChar (c : Char) : String :=
  if c.isAlphanum || c == '-' || c == '_' || c == '.' || c == '~' then
    c.toString
  else
    let n := c.toNat
    if n < 128 then
      s!"%{(n / 16).digitChar}{(n % 16).digitChar}"
    else
      -- For non-ASCII, encode as UTF-8 bytes
      let bytes := (String.singleton c).toUTF8
      String.join (bytes.toList.map fun b => s!"%{(b.toNat / 16).digitChar}{(b.toNat % 16).digitChar}")

/-- URL-encode a string -/
def urlEncode (s : String) : String :=
  String.join (s.toList.map encodeChar)

/-! ## HTTP Utilities -/

/-- Make HTTP GET request using curl -/
def httpGet (url : String) : IO String := do
  let output ← IO.Process.output {
    cmd := "curl"
    args := #["-s", "-L", "--max-time", "15", url]
  }
  if output.exitCode != 0 then
    throw (IO.userError s!"HTTP GET failed: {output.stderr}")
  return output.stdout

/-- Make HTTP POST request using curl -/
def httpPost (url : String) (body : String) (contentType : String := "application/json") : IO String := do
  let output ← IO.Process.output {
    cmd := "curl"
    args := #["-s", "-L", "--max-time", "15", "-X", "POST",
              "-H", s!"Content-Type: {contentType}",
              "-d", body, url]
  }
  if output.exitCode != 0 then
    throw (IO.userError s!"HTTP POST failed: {output.stderr}")
  return output.stdout

/-! ## Rate Limiting -/

/-- Simple in-memory rate limiter -/
structure RateLimiter where
  timestamps : IO.Ref (Array Nat)
  maxRequests : Nat
  perSeconds : Nat

def RateLimiter.new (maxRequests perSeconds : Nat) : IO RateLimiter := do
  let timestamps ← IO.mkRef #[]
  return { timestamps, maxRequests, perSeconds }

def RateLimiter.check (limiter : RateLimiter) : IO Bool := do
  let now ← IO.monoMsNow
  let cutoff := now - limiter.perSeconds * 1000
  let ts ← limiter.timestamps.get
  let valid := ts.filter (· > cutoff)
  limiter.timestamps.set valid
  return valid.size < limiter.maxRequests

def RateLimiter.record (limiter : RateLimiter) : IO Unit := do
  let now ← IO.monoMsNow
  limiter.timestamps.modify (·.push now)

/-! ## Result Helpers -/

def errorResult (s : String) : ToolResult :=
  { content := #[.text s], isError := true }

def jsonResult (j : Json) : ToolResult :=
  { content := #[.text j.compress], isError := false }

/-! ## LeanSearch -/

def leansearchToolDef : ToolDef := {
  name := "lean_leansearch"
  description := "Limit: 3req/30s. Search Mathlib via leansearch.net using natural language.\n\nExamples: \"sum of two even numbers is even\", \"Cauchy-Schwarz inequality\", \"{f : A → B} (hf : Injective f) : ∃ g, LeftInverse g f\""
  inputSchema := {
    properties := .mkObj [
      ("query", .mkObj [("type", .str "string"), ("description", .str "Natural language or Lean term query")]),
      ("num_results", .mkObj [("type", .str "integer"), ("description", .str "Max results (default 5)")])
    ]
    required := #["query"]
  }
}

def leansearchHandler (limiter : RateLimiter) (params : Json) : IO ToolResult := do
  let some query := params.getObjValAs? String "query" |>.toOption
    | return errorResult "Missing query"
  let numResults := params.getObjValAs? Nat "num_results" |>.toOption |>.getD 5

  unless ← limiter.check do
    return errorResult "Rate limit exceeded (3 requests per 30 seconds)"

  limiter.record

  let body := Json.mkObj [
    ("num_results", .str (toString numResults)),
    ("query", .arr #[.str query])
  ]
  let response ← httpPost "https://leansearch.net/search" body.compress

  match Json.parse response with
  | .error e => return errorResult s!"JSON parse error: {e}"
  | .ok json =>
    -- Response is array of results
    match json.getArr? with
    | .ok results =>
      let items := results.map fun r =>
        let name := r.getObjValAs? String "name" |>.toOption |>.getD ""
        let module := r.getObjValAs? String "module_name" |>.toOption |>.getD ""
        let kind := r.getObjValAs? String "kind" |>.toOption |>.getD ""
        let type := r.getObjValAs? String "type" |>.toOption |>.getD ""
        Json.mkObj [
          ("name", .str name),
          ("module", .str module),
          ("kind", .str kind),
          ("type", .str type)
        ]
      return jsonResult (.mkObj [("items", .arr items)])
    | .error _ =>
      return jsonResult (.mkObj [("items", .arr #[])])

/-! ## Loogle -/

def loogleToolDef : ToolDef := {
  name := "lean_loogle"
  description := "Search Mathlib by type signature via loogle.lean-lang.org.\n\nExamples: `Real.sin`, `\"comm\"`, `(?a → ?b) → List ?a → List ?b`, `_ * (_ ^ _)`, `|- _ < _ → _ + 1 < _ + 1`"
  inputSchema := {
    properties := .mkObj [
      ("query", .mkObj [("type", .str "string"), ("description", .str "Type pattern, constant, or name substring")]),
      ("num_results", .mkObj [("type", .str "integer"), ("description", .str "Max results (default 8)")])
    ]
    required := #["query"]
  }
}

def loogleHandler (limiter : RateLimiter) (params : Json) : IO ToolResult := do
  let some query := params.getObjValAs? String "query" |>.toOption
    | return errorResult "Missing query"
  let numResults := params.getObjValAs? Nat "num_results" |>.toOption |>.getD 8

  unless ← limiter.check do
    return errorResult "Rate limit exceeded (3 requests per 30 seconds)"

  limiter.record

  let encodedQuery := urlEncode query
  let url := s!"https://loogle.lean-lang.org/json?q={encodedQuery}"
  let response ← httpGet url

  match Json.parse response with
  | .error e => return errorResult s!"JSON parse error: {e}"
  | .ok json =>
    match json.getObjValAs? (Array Json) "hits" with
    | .ok hits =>
      let items := (hits.toSubarray 0 (min numResults hits.size)).toArray.map fun h =>
        let name := h.getObjValAs? String "name" |>.toOption |>.getD ""
        let type := h.getObjValAs? String "type" |>.toOption |>.getD ""
        let module := h.getObjValAs? String "module" |>.toOption |>.getD ""
        Json.mkObj [
          ("name", .str name),
          ("type", .str type),
          ("module", .str module)
        ]
      return jsonResult (.mkObj [("items", .arr items)])
    | .error _ =>
      -- Check for error message
      match json.getObjValAs? String "error" with
      | .ok err => return errorResult err
      | .error _ => return jsonResult (.mkObj [("items", .arr #[])])

/-! ## LeanFinder -/

def leanfinderToolDef : ToolDef := {
  name := "lean_leanfinder"
  description := "Limit: 10req/30s. Semantic search by mathematical meaning via Lean Finder.\n\nExamples: \"commutativity of addition on natural numbers\", \"I have h : n < m and need n + 1 < m + 1\", proof state text."
  inputSchema := {
    properties := .mkObj [
      ("query", .mkObj [("type", .str "string"), ("description", .str "Mathematical concept or proof state")]),
      ("num_results", .mkObj [("type", .str "integer"), ("description", .str "Max results (default 5)")])
    ]
    required := #["query"]
  }
}

def leanfinderHandler (limiter : RateLimiter) (params : Json) : IO ToolResult := do
  let some query := params.getObjValAs? String "query" |>.toOption
    | return errorResult "Missing query"
  let numResults := params.getObjValAs? Nat "num_results" |>.toOption |>.getD 5

  unless ← limiter.check do
    return errorResult "Rate limit exceeded (10 requests per 30 seconds)"

  limiter.record

  let body := Json.mkObj [
    ("query", .str query),
    ("k", .num numResults)
  ]
  let url := "https://bxrituxuhpc70w8w.us-east-1.aws.endpoints.huggingface.cloud"
  let response ← httpPost url body.compress

  match Json.parse response with
  | .error e => return errorResult s!"JSON parse error: {e}"
  | .ok json =>
    match json.getObjValAs? (Array Json) "results" with
    | .ok results =>
      let items := results.map fun r =>
        let formal := r.getObjValAs? String "formal_statement" |>.toOption |>.getD ""
        let informal := r.getObjValAs? String "informal_statement" |>.toOption |>.getD ""
        Json.mkObj [
          ("formal", .str formal),
          ("informal", .str informal)
        ]
      return jsonResult (.mkObj [("items", .arr items)])
    | .error _ =>
      return jsonResult (.mkObj [("items", .arr #[])])

/-! ## Local Search (ripgrep-based) -/

def localSearchToolDef : ToolDef := {
  name := "lean_local_search"
  description := "Fast local search to verify declarations exist. Use BEFORE trying a lemma name."
  inputSchema := {
    properties := .mkObj [
      ("query", .mkObj [("type", .str "string"), ("description", .str "Declaration name or prefix")]),
      ("limit", .mkObj [("type", .str "integer"), ("description", .str "Max matches (default 10)")]),
      ("project_root", .mkObj [("type", .str "string"), ("description", .str "Project root (inferred if omitted)")])
    ]
    required := #["query"]
  }
}

def localSearchHandler (params : Json) : IO ToolResult := do
  let some query := params.getObjValAs? String "query" |>.toOption
    | return errorResult "Missing query"
  let limit := params.getObjValAs? Nat "limit" |>.toOption |>.getD 10
  let projectRoot := params.getObjValAs? String "project_root" |>.toOption |>.getD "."

  -- Use ripgrep to search for declarations
  let pattern := s!"^\\s*(theorem|lemma|def|abbrev|class|instance|structure|inductive|opaque)\\s+[^\\s]*{query}"

  let output ← IO.Process.output {
    cmd := "rg"
    args := #[
      "--json", "--no-ignore", "--smart-case",
      "-g", "*.lean", "-g", "!.lake/build/**",
      "-m", toString limit,
      pattern, projectRoot
    ]
  }

  -- Parse ripgrep JSON output (one JSON object per line)
  let lines := output.stdout.splitOn "\n" |>.filter (·.trimAscii.toString.length > 0)
  let mut results : Array Json := #[]

  for line in lines do
    match Json.parse line with
    | .ok json =>
      let typeStr := json.getObjValAs? String "type" |>.toOption
      if typeStr == some "match" then
        match json.getObjVal? "data" with
        | .ok data =>
          let path := data.getObjVal? "path" |>.toOption.bind (·.getObjValAs? String "text" |>.toOption) |>.getD ""
          let lineData := data.getObjVal? "lines" |>.toOption.bind (·.getObjValAs? String "text" |>.toOption) |>.getD ""
          let lineNum := data.getObjValAs? Nat "line_number" |>.toOption |>.getD 0

          -- Extract declaration name
          let parts := lineData.trimAscii.toString.splitOn " "
          let kind := parts.head? |>.getD ""
          let name := (parts.getD 1 "" |>.takeWhile (· != ':') |>.takeWhile (· != ' ')).toString

          results := results.push (.mkObj [
            ("name", .str name),
            ("kind", .str kind),
            ("file", .str path),
            ("line", .num lineNum)
          ])
        | .error _ => pure ()
    | .error _ => pure ()

  return jsonResult (.mkObj [("items", .arr results)])

/-! ## Search Context -/

structure SearchContext where
  leansearchLimiter : RateLimiter
  loogleLimiter : RateLimiter
  leanfinderLimiter : RateLimiter

def SearchContext.new : IO SearchContext := do
  let leansearch ← RateLimiter.new 3 30
  let loogle ← RateLimiter.new 3 30
  let leanfinder ← RateLimiter.new 10 30
  return {
    leansearchLimiter := leansearch
    loogleLimiter := loogle
    leanfinderLimiter := leanfinder
  }

end LeanMcp.Search
