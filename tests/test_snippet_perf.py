"""Compare old (collectAxioms) vs new (project-local BFS) goal_tracker snippets.

Run with:
    .venv/bin/python tests/test_snippet_perf.py
"""

import json
import time
from pathlib import Path

from leanclient import LeanLSPClient, DocumentContentChange


PROJECT = Path("/home/maor/Desktop/git/QuantumInformation/QuantumInformation")
FILE = "QuantumInformation/InfoTheory/DeFinetti/Theorem/Main.lean"
DECL = "InfoTheory.DeFinetti.quantum_deFinetti"


def make_snippet_old(decl_name: str) -> str:
    """Current approach: collectAxioms as oracle."""
    return f"""
open Lean Lean.Elab.Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mkN (n : Name) (s : String) : Name := if let some k := s.toNat? then n.num k else n.str s
  let target : Name := "{decl_name}".splitOn "." |>.foldl mkN .anonymous
  if (env.find? target).isNone then throwError "not found: {decl_name}"
  let targetAxioms ← Lean.collectAxioms target
  if !targetAxioms.contains ``sorryAx then
    let summary := Json.mkObj [("visited", Json.num 1)]
    logInfo m!"MCP_SUMMARY:{{summary.compress}}"
  else
    let mut visited : NameSet := .empty
    let mut queue : Array Name := #[target]
    let mut nodeInfo : Array (Name × Bool × Array Name) := #[]
    let mut axCache : PersistentHashMap Name (Array Name) := .empty
    axCache := axCache.insert target targetAxioms
    while queue.size > 0 do
      let name := queue.back!
      queue := queue.pop
      if visited.contains name then continue
      visited := visited.insert name
      if (env.find? name).isNone then continue
      let ci := (env.find? name).get!
      let allConsts := ci.getUsedConstantsAsSet
      let explicit := allConsts.contains ``sorryAx
      let mut sorryDeps : Array Name := #[]
      for dep in allConsts.toArray do
        if dep == ``sorryAx then continue
        if (env.find? dep).isNone then continue
        let depAxioms ← match axCache.find? dep with
          | some ax => pure ax
          | none => do
            let ax ← Lean.collectAxioms dep
            axCache := axCache.insert dep ax
            pure ax
        if depAxioms.contains ``sorryAx then
          sorryDeps := sorryDeps.push dep
          if !visited.contains dep then
            queue := queue.push dep
      nodeInfo := nodeInfo.push (name, explicit, sorryDeps)
    for (name, explicit, sorryDeps) in nodeInfo do
      let mut fields : Array (String × Json) := #[
        ("name", Json.str name.toString),
        ("explicit", Json.bool explicit),
        ("sorry_deps", Json.arr (sorryDeps.map fun d => Json.str d.toString))
      ]
      if explicit then
        match env.getModuleFor? name with
        | some mod => fields := fields.push ("module", Json.str mod.toString)
        | none => pure ()
        match ← findDeclarationRanges? name with
        | some ranges => fields := fields.push ("line", Json.num ranges.range.pos.line)
        | none => pure ()
      logInfo m!"MCP_NODE:{{(Json.mkObj fields.toList).compress}}"
    let summary := Json.mkObj [("visited", Json.num visited.size)]
    logInfo m!"MCP_SUMMARY:{{summary.compress}}"
"""


def make_snippet_new(decl_name: str) -> str:
    """New approach: BFS only through project-local deps, no collectAxioms."""
    return f"""
open Lean Lean.Elab.Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mkN (n : Name) (s : String) : Name := if let some k := s.toNat? then n.num k else n.str s
  let target : Name := "{decl_name}".splitOn "." |>.foldl mkN .anonymous
  if (env.find? target).isNone then throwError "not found: {decl_name}"
  -- Helper: check if a dep is from an external package
  let isExternal (dep : Name) : Bool :=
    match env.getModuleFor? dep with
    | some mod =>
      let root := mod.getRoot.toString
      root == "Mathlib" || root == "Init" || root == "Lean" || root == "Std" ||
      root == "Batteries" || root == "Aesop" || root == "ProofWidgets" ||
      root == "ImportGraph" || root == "LabelAttr" || root == "Qq" ||
      root == "Cache"
    | none => false
  -- Phase 1: BFS through project-local deps only
  let mut visited : NameSet := .empty
  let mut queue : Array Name := #[target]
  let mut nodeMap : PersistentHashMap Name (Bool × Array Name) := .empty
  while queue.size > 0 do
    let name := queue.back!
    queue := queue.pop
    if visited.contains name then continue
    visited := visited.insert name
    if (env.find? name).isNone then continue
    let ci := (env.find? name).get!
    let allConsts := ci.getUsedConstantsAsSet
    let explicit := allConsts.contains ``sorryAx
    let mut localDeps : Array Name := #[]
    for dep in allConsts.toArray do
      if dep == ``sorryAx then continue
      if (env.find? dep).isNone then continue
      if isExternal dep then continue
      localDeps := localDeps.push dep
      if !visited.contains dep then
        queue := queue.push dep
    nodeMap := nodeMap.insert name (explicit, localDeps)
  -- Phase 2: Propagate sorry-taint (fixed point)
  let mut tainted : NameSet := .empty
  for name in visited.toArray do
    match nodeMap.find? name with
    | some (true, _) => tainted := tainted.insert name
    | _ => pure ()
  let mut changed := true
  while changed do
    changed := false
    for name in visited.toArray do
      if tainted.contains name then continue
      match nodeMap.find? name with
      | some (_, deps) =>
        for dep in deps do
          if tainted.contains dep then
            tainted := tainted.insert name
            changed := true
            break
      | none => pure ()
  -- Emit results
  if !tainted.contains target then
    let summary := Json.mkObj [("visited", Json.num 1)]
    logInfo m!"MCP_SUMMARY:{{summary.compress}}"
  else
    for name in visited.toArray do
      if !tainted.contains name then continue
      let (explicit, deps) := match nodeMap.find? name with
        | some v => v
        | none => (false, #[])
      let sorryDeps := deps.filter (tainted.contains ·)
      let mut fields : Array (String × Json) := #[
        ("name", Json.str name.toString),
        ("explicit", Json.bool explicit),
        ("sorry_deps", Json.arr (sorryDeps.map fun d => Json.str d.toString))
      ]
      if explicit then
        match env.getModuleFor? name with
        | some mod => fields := fields.push ("module", Json.str mod.toString)
        | none => pure ()
        match ← findDeclarationRanges? name with
        | some ranges => fields := fields.push ("line", Json.num ranges.range.pos.line)
        | none => pure ()
      logInfo m!"MCP_NODE:{{(Json.mkObj fields.toList).compress}}"
    let summary := Json.mkObj [("visited", Json.num tainted.size)]
    logInfo m!"MCP_SUMMARY:{{summary.compress}}"
"""


def parse_results(diagnostics: list[dict]) -> dict:
    """Parse MCP_NODE/MCP_SUMMARY lines from diagnostics."""
    nodes = {}
    visited = 0
    for diag in diagnostics:
        if diag.get("severity") != 3:
            continue
        msg = diag.get("message", "")
        if msg.startswith("MCP_NODE:"):
            obj = json.loads(msg[len("MCP_NODE:") :])
            nodes[obj["name"]] = obj
        elif msg.startswith("MCP_SUMMARY:"):
            obj = json.loads(msg[len("MCP_SUMMARY:") :])
            visited = obj.get("visited", 0)
    return {"nodes": nodes, "visited": visited}


def run_snippet(
    client: LeanLSPClient, rel_path: str, snippet: str
) -> tuple[dict, float]:
    """Append snippet, get diagnostics, restore, return (parsed_result, elapsed_seconds)."""
    original = client.get_file_content(rel_path)
    original_lines = original.split("\n")
    appended_line = len(original_lines)
    snippet_lines = snippet.count("\n")

    change = DocumentContentChange(snippet, [appended_line, 0], [appended_line, 0])
    client.update_file(rel_path, [change])

    t0 = time.monotonic()
    raw = client.get_diagnostics(
        rel_path, start_line=appended_line, inactivity_timeout=300.0
    )
    elapsed = time.monotonic() - t0

    diags = list(raw)

    # Check for errors
    errors = [d.get("message", "") for d in diags if d.get("severity") == 1]
    if errors:
        print(f"  ERRORS: {errors[:3]}")

    result = parse_results(diags)

    # Restore
    restore = DocumentContentChange(
        "", [appended_line, 0], [appended_line + snippet_lines, 0]
    )
    client.update_file(rel_path, [restore])

    return result, elapsed


def compare_results(old_res: dict, new_res: dict):
    """Compare the sorry leaves found by both approaches."""
    old_sorry = {n for n, obj in old_res["nodes"].items() if obj.get("explicit")}
    new_sorry = {n for n, obj in new_res["nodes"].items() if obj.get("explicit")}

    old_tainted = set(old_res["nodes"].keys())
    new_tainted = set(new_res["nodes"].keys())

    print(
        f"\n  Old: {len(old_sorry)} sorry leaves, {len(old_tainted)} tainted nodes, {old_res['visited']} visited"
    )
    print(
        f"  New: {len(new_sorry)} sorry leaves, {len(new_tainted)} tainted nodes, {new_res['visited']} visited"
    )

    if old_sorry == new_sorry:
        print("  Sorry leaves MATCH")
    else:
        print(f"  MISMATCH! Only in old: {old_sorry - new_sorry}")
        print(f"  MISMATCH! Only in new: {new_sorry - old_sorry}")

    if old_tainted == new_tainted:
        print("  Tainted nodes MATCH")
    else:
        only_old = old_tainted - new_tainted
        only_new = new_tainted - old_tainted
        if only_old:
            print(f"  Only in old ({len(only_old)}): {list(only_old)[:5]}...")
        if only_new:
            print(f"  Only in new ({len(only_new)}): {list(only_new)[:5]}...")


def main():
    print(f"Project: {PROJECT}")
    print(f"File: {FILE}")
    print(f"Decl: {DECL}")

    client = LeanLSPClient(str(PROJECT))
    try:
        print("\nOpening file...")
        client.open_file(FILE)
        # Wait for initial elaboration
        print("Waiting for initial elaboration...")
        client.get_diagnostics(FILE, inactivity_timeout=300.0)
        print("File ready.\n")

        # Run old snippet
        print("Running OLD snippet (collectAxioms)...")
        old_res, old_time = run_snippet(client, FILE, make_snippet_old(DECL))
        print(f"  Time: {old_time:.1f}s")

        # Wait a bit for restore to settle
        print("\nWaiting for restore to settle...")
        client.get_diagnostics(FILE, inactivity_timeout=60.0)

        # Run new snippet
        print("\nRunning NEW snippet (project-local BFS)...")
        new_res, new_time = run_snippet(client, FILE, make_snippet_new(DECL))
        print(f"  Time: {new_time:.1f}s")

        # Compare
        print("\n=== COMPARISON ===")
        print(f"  Old: {old_time:.1f}s")
        print(f"  New: {new_time:.1f}s")
        print(f"  Speedup: {old_time / new_time:.1f}x" if new_time > 0 else "  N/A")
        compare_results(old_res, new_res)

    finally:
        client.close()


if __name__ == "__main__":
    main()
