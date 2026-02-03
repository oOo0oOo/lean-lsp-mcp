"""Header/body splitting for REPL caching."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitResult:
    header: str
    body: str


def split_code(code: str) -> SplitResult:
    """Split code into header (imports) and body."""
    lines = code.splitlines()

    i = 0
    while i < len(lines) and (
        not lines[i].strip() or lines[i].strip().startswith("import ")
    ):
        i += 1

    header_lines = [ln.strip() for ln in lines[:i] if ln.strip().startswith("import ")]

    # Consolidate Mathlib imports
    has_mathlib = any(ln.startswith("import Mathlib") for ln in header_lines)
    other_imports = [ln for ln in header_lines if not ln.startswith("import Mathlib")]

    result = []
    if has_mathlib:
        result.append("import Mathlib")
    result.extend(dict.fromkeys(other_imports))  # dedupe preserving order

    return SplitResult(
        header="\n".join(result),
        body="\n".join(lines[i:]),
    )
