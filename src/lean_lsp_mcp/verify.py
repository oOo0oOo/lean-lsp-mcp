"""Theorem verification: axiom checking + source scanning."""

from __future__ import annotations

import re
import subprocess
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


# The three axioms all of Mathlib rests on. Anything else changes what a
# `#print axioms` result means, and the flat list alone does not say so.
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

# `sorry` leaves this behind: the theorem is not proved at all.
_INCOMPLETE_AXIOMS = frozenset({"sorryAx"})

# `native_decide` discharges a goal by running compiled code, so the result
# rests on the compiler and the runtime rather than on the kernel.
_NATIVE_AXIOMS = frozenset({"Lean.ofReduceBool", "Lean.trustCompiler"})
# Recent Lean versions generate a per-declaration axiom for native_decide
# (Lean.Meta.nativeEqTrue) instead of using the older global axiom.
_NATIVE_DECIDE_AXIOM = re.compile(r"(?:^|\.)_native\.native_decide\.ax(?:_\d+)*$")


def classify_axioms(axioms: list[str]) -> tuple[str, list[str]]:
    """Summarise an axiom list as a trust verdict plus the axioms behind it.

    The verdict is the worst case present, because that is what bounds what
    the theorem is worth: an incomplete proof cannot be redeemed by the rest
    of the list being ordinary.
    """
    non_standard = [axiom for axiom in axioms if axiom not in STANDARD_AXIOMS]

    if any(axiom in _INCOMPLETE_AXIOMS for axiom in axioms):
        return "incomplete", non_standard
    if any(
        axiom in _NATIVE_AXIOMS or _NATIVE_DECIDE_AXIOM.search(axiom)
        for axiom in axioms
    ):
        return "native", non_standard
    if non_standard:
        return "custom", non_standard
    return "standard", non_standard


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
            encoding="utf-8",
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
