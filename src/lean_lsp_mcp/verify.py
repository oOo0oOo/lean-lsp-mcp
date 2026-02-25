"""Theorem verification: axiom checking + source scanning."""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from orjson import loads as _json_loads

# Patterns that may affect soundness - all warnings, LLM decides risk
_WARNING_PATTERNS: list[str] = [
    r"set_option\s+debug\.",
    r"\bunsafe\b",
    r"@\[implemented_by\b",
    r"@\[extern\b",
    r"\bopaque\b",
    r"local\s+instance\b",
    r"local\s+notation\b",
    r"local\s+macro_rules\b",
    r"scoped\s+notation\b",
    r"scoped\s+instance\b",
    r"@\[csimp\b",
    r"import\s+Lean\.Elab\b",
    r"import\s+Lean\.Meta\b",
]

_COMBINED_PATTERN = "|".join(f"(?:{p})" for p in _WARNING_PATTERNS)


def _cleanup_stale_verify_files(project_path: Path) -> None:
    for f in project_path.glob("_mcp_verify_*.lean"):
        try:
            f.unlink()
        except OSError:
            pass


def make_axiom_check(
    file_path: Path, project_path: Path, theorem_name: str
) -> tuple[str, Path]:
    """Create temp file for axiom checking. Returns (rel_path, abs_path)."""
    _cleanup_stale_verify_files(project_path)
    rel = file_path.resolve().relative_to(project_path.resolve())
    module = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
    rel_path = f"_mcp_verify_{uuid.uuid4().hex}.lean"
    tmp = project_path / rel_path
    tmp.write_text(f"import {module}\n#print axioms {theorem_name}\n", encoding="utf-8")
    return rel_path, tmp


def parse_axioms(diagnostics: list[dict]) -> list[str]:
    """Extract axiom names from #print axioms info diagnostics."""
    axioms: list[str] = []
    for diag in diagnostics:
        if diag.get("severity") != 3:  # info
            continue
        msg = diag.get("message", "").replace("\n", " ")
        if m := re.search(r"depends on axioms:\s*\[(.+?)\]", msg):
            axioms.extend(a.strip() for a in m.group(1).split(","))
    return axioms


def check_axiom_errors(diagnostics: list[dict]) -> str | None:
    """Return joined error messages if any, else None."""
    errors = [d.get("message", "") for d in diagnostics if d.get("severity") == 1]
    return "; ".join(errors) if errors else None


def scan_warnings(file_path: Path) -> list[dict[str, int | str]]:
    """Scan file for suspicious patterns via rg. Returns [{line, pattern}]."""
    try:
        proc = subprocess.run(
            [
                "rg",
                "--json",
                "--no-ignore",
                "--no-messages",
                _COMBINED_PATTERN,
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    warnings: list[dict[str, int | str]] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        event = _json_loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        text = data["lines"]["text"].strip()
        for pattern in _WARNING_PATTERNS:
            if m := re.search(pattern, text):
                warnings.append({"line": data["line_number"], "pattern": m.group(0)})
                break
    return warnings
