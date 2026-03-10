"""Compare current (project-local BFS) vs optimized (module-filtered BFS) goal_tracker snippets.

The optimized version:
1. Greps project files for `sorry` to find sorry-containing modules (instant)
2. Parses import statements to build module graph (instant)
3. Finds which modules transitively import a sorry-containing module (instant)
4. Passes that allowlist into the Lean snippet so the BFS skips unrelated modules

Run with:
    .venv/bin/python tests/test_snippet_perf2.py
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from leanclient import LeanLSPClient, DocumentContentChange


PROJECT = Path("/home/maor/Desktop/git/QuantumInformation/QuantumInformation")
SRC_ROOT = PROJECT / "QuantumInformation"
FILE = "QuantumInformation/InfoTheory/DeFinetti/Theorem/Main.lean"
DECL = "InfoTheory.DeFinetti.quantum_deFinetti"

# Known external module roots (skip these entirely)
EXTERNAL_ROOTS = {
    "Mathlib", "Init", "Lean", "Std", "Batteries", "Aesop",
    "ProofWidgets", "ImportGraph", "LabelAttr", "Qq", "Cache",
}


def file_to_module(file_path: Path, src_root: Path) -> str:
    """Convert a file path to a Lean module name."""
    rel = file_path.relative_to(src_root.parent)
    return str(rel).removesuffix(".lean").replace("/", ".")


def find_sorry_modules(src_root: Path) -> set[str]:
    """Grep for sorry in project files, return module names."""
    sorry_modules = set()
    for lean_file in src_root.rglob("*.lean"):
        text = lean_file.read_text()
        # Match 'sorry' as a word (not in comments ideally, but good enough)
        if re.search(r'\bsorry\b', text):
            sorry_modules.add(file_to_module(lean_file, src_root))
    return sorry_modules


def build_import_graph(src_root: Path) -> dict[str, set[str]]:
    """Parse import statements to build module -> set of imported modules."""
    graph = defaultdict(set)
    for lean_file in src_root.rglob("*.lean"):
        mod = file_to_module(lean_file, src_root)
        for line in lean_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("import "):
                imported = line[len("import "):].strip()
                graph[mod].add(imported)
    return dict(graph)


def find_sorry_reachable_modules(
    import_graph: dict[str, set[str]],
    sorry_modules: set[str],
    all_modules: set[str],
) -> set[str]:
    """Find all modules that transitively import a sorry-containing module.

    A module is sorry-reachable if:
    - It contains sorry itself, OR
    - It imports (transitively) a sorry-containing module
    """
    # Build reverse graph: module -> set of modules that import it
    reverse_graph = defaultdict(set)
    for mod, imports in import_graph.items():
        for imp in imports:
            reverse_graph[imp].add(mod)

    # BFS from sorry modules upward through reverse imports
    reachable = set(sorry_modules)
    queue = list(sorry_modules)
    while queue:
        mod = queue.pop()
        for importer in reverse_graph.get(mod, []):
            if importer not in reachable:
                reachable.add(importer)
                queue.append(importer)

    return reachable


def make_snippet_current(decl_name: str) -> str:
    """Current approach: project-local BFS, no module filtering."""
    return f"""
open Lean Lean.Elab.Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mkN (n : Name) (s : String) : Name := if let some k := s.toNat? then n.num k else n.str s
  let target : Name := "{decl_name}".splitOn "." |>.foldl mkN .anonymous
  if (env.find? target).isNone then throwError "not found: {decl_name}"
  let isExternal (dep : Name) : Bool :=
    match env.getModuleFor? dep with
    | some mod =>
      let root := mod.getRoot.toString
      root == "Mathlib" || root == "Init" || root == "Lean" || root == "Std" ||
      root == "Batteries" || root == "Aesop" || root == "ProofWidgets" ||
      root == "ImportGraph" || root == "LabelAttr" || root == "Qq" ||
      root == "Cache"
    | none => false
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


def make_snippet_filtered(decl_name: str, allowed_modules: set[str]) -> str:
    """Optimized: BFS restricted to allowed modules only."""
    # Build Lean array literal for allowed module names
    mod_array = ", ".join(
        f'"{m}".splitOn "." |>.foldl mkN .anonymous'
        for m in sorted(allowed_modules)
    )

    return f"""
open Lean Lean.Elab.Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mkN (n : Name) (s : String) : Name := if let some k := s.toNat? then n.num k else n.str s
  let target : Name := "{decl_name}".splitOn "." |>.foldl mkN .anonymous
  if (env.find? target).isNone then throwError "not found: {decl_name}"
  -- Build allowed module set
  let modList : Array Name := #[{mod_array}]
  let mut allowedMods : PersistentHashMap Name Bool := .empty
  for m in modList do
    allowedMods := allowedMods.insert m true
  -- Helper: check if dep is in an allowed module
  let isAllowed (dep : Name) : Bool :=
    match env.getModuleFor? dep with
    | some mod => (allowedMods.find? mod).isSome
    | none => true  -- current file, always check
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
      if !isAllowed dep then continue
      localDeps := localDeps.push dep
      if !visited.contains dep then
        queue := queue.push dep
    nodeMap := nodeMap.insert name (explicit, localDeps)
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
            obj = json.loads(msg[len("MCP_NODE:"):])
            nodes[obj["name"]] = obj
        elif msg.startswith("MCP_SUMMARY:"):
            obj = json.loads(msg[len("MCP_SUMMARY:"):])
            visited = obj.get("visited", 0)
    return {"nodes": nodes, "visited": visited}


def run_snippet(client: LeanLSPClient, rel_path: str, snippet: str) -> tuple[dict, float]:
    """Append snippet, get diagnostics, restore, return (parsed_result, elapsed_seconds)."""
    original = client.get_file_content(rel_path)
    original_lines = original.split("\n")
    appended_line = len(original_lines)
    snippet_lines = snippet.count("\n")

    change = DocumentContentChange(snippet, [appended_line, 0], [appended_line, 0])
    client.update_file(rel_path, [change])

    t0 = time.monotonic()
    raw = client.get_diagnostics(rel_path, start_line=appended_line, inactivity_timeout=300.0)
    elapsed = time.monotonic() - t0

    diags = list(raw)

    errors = [d.get("message", "") for d in diags if d.get("severity") == 1]
    if errors:
        print(f"  ERRORS: {errors[:3]}")

    result = parse_results(diags)

    restore = DocumentContentChange("", [appended_line, 0], [appended_line + snippet_lines, 0])
    client.update_file(rel_path, [restore])

    return result, elapsed


def compare_results(cur_res: dict, filt_res: dict):
    """Compare sorry leaves found by both approaches."""
    cur_sorry = {n for n, obj in cur_res["nodes"].items() if obj.get("explicit")}
    filt_sorry = {n for n, obj in filt_res["nodes"].items() if obj.get("explicit")}

    cur_tainted = set(cur_res["nodes"].keys())
    filt_tainted = set(filt_res["nodes"].keys())

    print(f"\n  Current:  {len(cur_sorry)} sorry leaves, {len(cur_tainted)} tainted, {cur_res['visited']} visited")
    print(f"  Filtered: {len(filt_sorry)} sorry leaves, {len(filt_tainted)} tainted, {filt_res['visited']} visited")

    if cur_sorry == filt_sorry:
        print("  Sorry leaves MATCH")
    else:
        print(f"  MISMATCH! Only in current: {cur_sorry - filt_sorry}")
        print(f"  MISMATCH! Only in filtered: {filt_sorry - cur_sorry}")

    if cur_tainted == filt_tainted:
        print("  Tainted nodes MATCH")
    else:
        only_cur = cur_tainted - filt_tainted
        only_filt = filt_tainted - cur_tainted
        if only_cur:
            print(f"  Only in current ({len(only_cur)}): {list(only_cur)[:5]}...")
        if only_filt:
            print(f"  Only in filtered ({len(only_filt)}): {list(only_filt)[:5]}...")


def main():
    print(f"Project: {PROJECT}")
    print(f"File: {FILE}")
    print(f"Decl: {DECL}\n")

    # Phase 0: Pre-compute module filter (instant)
    t0 = time.monotonic()
    sorry_modules = find_sorry_modules(SRC_ROOT)
    import_graph = build_import_graph(SRC_ROOT)
    all_modules = set(import_graph.keys()) | {
        file_to_module(f, SRC_ROOT) for f in SRC_ROOT.rglob("*.lean")
    }
    reachable = find_sorry_reachable_modules(import_graph, sorry_modules, all_modules)
    precompute_time = time.monotonic() - t0

    print(f"Pre-computation: {precompute_time*1000:.0f}ms")
    print(f"  Sorry-containing modules: {len(sorry_modules)}")
    print(f"  Total project modules: {len(all_modules)}")
    print(f"  Sorry-reachable modules: {len(reachable)}")
    print(f"  Modules pruned: {len(all_modules) - len(reachable)}")
    print()

    client = LeanLSPClient(str(PROJECT))
    try:
        print("Opening file...")
        client.open_file(FILE)
        print("Waiting for initial elaboration...")
        client.get_diagnostics(FILE, inactivity_timeout=300.0)
        print("File ready.\n")

        # Run current snippet
        print("Running CURRENT snippet (project-local BFS)...")
        cur_res, cur_time = run_snippet(client, FILE, make_snippet_current(DECL))
        print(f"  Time: {cur_time:.1f}s")

        # Wait for restore
        print("\nWaiting for restore...")
        client.get_diagnostics(FILE, inactivity_timeout=60.0)

        # Run filtered snippet
        print("\nRunning FILTERED snippet (module-filtered BFS)...")
        filt_res, filt_time = run_snippet(client, FILE, make_snippet_filtered(DECL, reachable))
        print(f"  Time: {filt_time:.1f}s")

        # Compare
        print("\n=== COMPARISON ===")
        print(f"  Current:  {cur_time:.1f}s")
        print(f"  Filtered: {filt_time:.1f}s  (+ {precompute_time*1000:.0f}ms precompute)")
        if filt_time > 0:
            print(f"  Speedup: {cur_time / filt_time:.1f}x")
        compare_results(cur_res, filt_res)

    finally:
        client.close()


if __name__ == "__main__":
    main()
