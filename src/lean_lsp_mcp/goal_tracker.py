"""Goal tracker: find sorry dependencies in a declaration's transitive closure.

Mirrors the tree output of QuantumInformation/scripts/goal_tracker.py but
runs entirely via LSP (no ExportDecls, no oleans needed).  The Lean
``#eval`` block BFS-walks only **project-local** dependencies (skipping
Mathlib/Init/Std/etc.), then propagates sorry-taint bottom-up to identify
which declarations transitively depend on sorry.

Key optimisation: instead of using ``collectAxioms`` (which walks into
Mathlib and is very slow), the BFS only follows constants whose module
is from the user's project.  Since external libraries are sorry-free,
any sorry must flow through project code.  This keeps the search fast
(~6s vs ~74s on a large project).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Lean snippet
# ---------------------------------------------------------------------------


def make_sorry_snippet(decl_name: str) -> str:
    """Return a ``#eval`` block that BFS-walks sorry-tainted deps.

    Instead of using ``collectAxioms`` (which walks into Mathlib and is
    slow), this only follows **project-local** dependencies — constants
    whose module is not from Mathlib/Init/Std/etc.  Since external
    libraries are sorry-free, any sorry must flow through project code.

    Phase 1: BFS through project-local deps, recording each node's
    direct children and whether it directly uses ``sorryAx``.

    Phase 2: Fixed-point propagation to mark all nodes that
    transitively depend on sorry.

    Output format (one ``logInfo`` per sorry-tainted node)::

        MCP_NODE:<json>

    where *json* is ``{{"name":..., "explicit":bool, "sorry_deps":[...],
    "module":str|null, "line":int|null}}``.  ``module`` and ``line`` are
    emitted for nodes with ``explicit=true`` so callers can locate sorry
    leaves without a separate search.

    A final summary line is emitted as::

        MCP_SUMMARY:<json>

    with ``{{"visited":int}}``.
    """
    return f"""
set_option maxHeartbeats 200000 in
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


# ---------------------------------------------------------------------------
# Parse diagnostics
# ---------------------------------------------------------------------------


@dataclass
class SorryNode:
    """One declaration in the sorry-issue tree."""

    name: str
    explicit_sorry: bool = False
    sorry_deps: list[str] = field(default_factory=list)
    module: str | None = None
    line: int | None = None


def parse_sorry_result(diagnostics: list[dict]) -> tuple[dict[str, SorryNode], int]:
    """Parse MCP_NODE / MCP_SUMMARY lines from diagnostics.

    Returns (nodes_by_name, total_visited).
    """
    nodes: dict[str, SorryNode] = {}
    visited = 0

    for diag in diagnostics:
        if diag.get("severity") != 3:  # info
            continue
        msg = diag.get("message", "")
        if msg.startswith("MCP_NODE:"):
            raw = msg[len("MCP_NODE:") :]
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            name = obj["name"]
            line_num = obj.get("line")
            nodes[name] = SorryNode(
                name=name,
                explicit_sorry=obj.get("explicit", False),
                sorry_deps=obj.get("sorry_deps", []),
                module=obj.get("module"),
                line=int(line_num) if line_num is not None else None,
            )
        elif msg.startswith("MCP_SUMMARY:"):
            raw = msg[len("MCP_SUMMARY:") :]
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            visited = obj.get("visited", 0)

    return nodes, visited


# ---------------------------------------------------------------------------
# Tree rendering  (mirrors goal_tracker.py print_issue_tree)
# ---------------------------------------------------------------------------


def render_tree(target: str, nodes: dict[str, SorryNode]) -> list[str]:
    """Render an ASCII dependency tree rooted at *target*.

    Only branches that lead to sorry are shown.  Each node is annotated
    with ``[explicit sorry]`` when the declaration itself contains
    ``sorryAx``.
    """
    lines: list[str] = []
    visited: set[str] = set()

    def _walk(name: str, prefix: str, is_last: bool) -> None:
        if name in visited:
            connector = "└─ " if is_last else "├─ "
            lines.append(f"{prefix}{connector}{name} (see above)")
            return
        visited.add(name)

        node = nodes.get(name)
        connector = "└─ " if is_last else "├─ "
        tag = " [explicit sorry]" if node and node.explicit_sorry else ""
        lines.append(f"{prefix}{connector}{name}{tag}")

        if node is None:
            return

        children = [d for d in node.sorry_deps if d in nodes]
        extension = "   " if is_last else "│  "
        new_prefix = prefix + extension
        for i, child in enumerate(children):
            _walk(child, new_prefix, i == len(children) - 1)

    _walk(target, "", True)
    return lines
