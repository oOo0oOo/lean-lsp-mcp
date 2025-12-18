"""Declaration extraction from Lean 4 source files.

This module provides regex-based extraction of declarations from Lean files.
For more accurate extraction from mathlib, see training_data.py which uses
the lean-training-data metaprogramming tools.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from .models import LeanDeclaration

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute hash of file contents for change detection."""
    try:
        return hashlib.md5(file_path.read_bytes()).hexdigest()[:12]
    except Exception:
        return ""


def infer_module_name(file_path: Path, base_path: Path | None = None) -> str:
    """Infer module name from file path.

    Handles common Lean project structures:
    - Mathlib/Algebra/Group/Basic.lean -> Mathlib.Algebra.Group.Basic
    - .lake/packages/foo/src/Foo/Bar.lean -> Foo.Bar
    - src/MyLib/Foo.lean -> MyLib.Foo
    - Basic.lean -> Basic
    """
    parts = file_path.parts
    path_str = str(file_path)

    # Handle .lake/packages paths specially
    # For packages, return empty module name - let the namespace handle naming
    # This avoids duplicating directory structure with Lean namespaces
    if ".lake/packages/" in path_str or ".lake\\packages\\" in path_str:
        return ""

    # Try to find a recognizable root (standard libraries)
    roots = {"Mathlib", "Std", "Init", "Lean", "Batteries", "Aesop", "ProofWidgets"}

    for i, part in enumerate(parts):
        if part in roots:
            # Found a known library root
            rel_parts = parts[i:]
            if rel_parts[-1].endswith(".lean"):
                rel_parts = rel_parts[:-1] + (rel_parts[-1][:-5],)
            return ".".join(rel_parts)

    # Try relative to base_path if provided
    if base_path:
        try:
            rel = file_path.relative_to(base_path)
            rel_parts = rel.parts
            if rel_parts[-1].endswith(".lean"):
                rel_parts = rel_parts[:-1] + (rel_parts[-1][:-5],)
            # Skip "src" directory
            if rel_parts and rel_parts[0] == "src":
                rel_parts = rel_parts[1:]
            return ".".join(rel_parts)
        except ValueError:
            pass

    # Fallback: just use stem
    return file_path.stem


def find_lean_files(
    root: Path, exclude_build: bool = True, max_files: int | None = None
) -> list[Path]:
    """Find all .lean files under root, excluding build directories."""
    files = []
    exclude_patterns = {
        ".lake/build",
        ".lake/packages/.lake",
        "__pycache__",
        ".git",
        "lake-packages",  # Old lake format
    }

    try:
        for lean_file in root.rglob("*.lean"):
            path_str = str(lean_file)
            if exclude_build and any(ex in path_str for ex in exclude_patterns):
                continue
            files.append(lean_file)
            if max_files and len(files) >= max_files:
                break
    except PermissionError:
        logger.warning(f"Permission denied accessing {root}")

    return files


def extract_declarations_from_file(
    file_path: Path, module_prefix: str = "", base_path: Path | None = None
) -> list[LeanDeclaration]:
    """Extract declarations from a single Lean file.

    Uses regex patterns optimized for common Lean 4 declaration styles.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Could not read {file_path}: {e}")
        return []

    declarations: list[LeanDeclaration] = []

    # Determine module name
    module_name = module_prefix or infer_module_name(file_path, base_path)

    # Track current namespace for qualified names
    namespace_stack: list[str] = []

    # Find namespace declarations
    namespace_pattern = re.compile(r"^namespace\s+([\w\.]+)", re.MULTILINE)
    end_pattern = re.compile(r"^end\s+([\w\.]+)?", re.MULTILINE)

    # Build namespace context map (line -> active namespace)
    lines = content.split("\n")
    namespace_at_line: dict[int, str] = {}
    current_ns = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if ns_match := namespace_pattern.match(stripped):
            namespace_stack.append(ns_match.group(1))
            current_ns = ".".join(namespace_stack)
        elif end_pattern.match(stripped):
            if namespace_stack:
                namespace_stack.pop()
                current_ns = ".".join(namespace_stack)
        namespace_at_line[i] = current_ns

    # Pattern to capture declarations with docstrings and signatures
    # Handles multi-line signatures by matching until := or where or |
    decl_pattern = re.compile(
        r"(?P<docstring>/--[\s\S]*?-/\s*)?"
        r"(?:@\[[\w\s,\(\)=\"\'\.]+\]\s*)*"  # attributes
        r"(?:private\s+|protected\s+|scoped\s+)?"
        r"(?:noncomputable\s+|unsafe\s+|partial\s+|nonrec\s+)*"
        r"(?P<kind>theorem|lemma|def|abbrev|class|structure|inductive|instance|axiom|opaque)\s+"
        r"(?P<name>[\w']+)"  # Just the base name, not qualified
        r"(?P<params>(?:\s*[\[\(\{][\s\S]*?[\]\)\}])*)"  # Parameters
        r"(?:\s*:\s*(?P<type>[^:=\n]+?))?"  # Optional type annotation
        r"(?=\s*(?::=|where|:|\||$))",  # Look ahead for definition start
        re.MULTILINE,
    )

    for match in decl_pattern.finditer(content):
        kind = match.group("kind")
        base_name = match.group("name")
        docstring = match.group("docstring")
        params = match.group("params") or ""
        type_ann = match.group("type") or ""

        # Clean up docstring
        if docstring:
            docstring = docstring.strip()
            if docstring.startswith("/--"):
                docstring = docstring[3:]
            if docstring.endswith("-/"):
                docstring = docstring[:-2]
            docstring = " ".join(docstring.split())  # Normalize whitespace

        # Build signature from params and type
        sig_parts = []
        if params:
            sig_parts.append(params.strip())
        if type_ann:
            sig_parts.append(f": {type_ann.strip()}")
        signature = " ".join(sig_parts)

        # Calculate line number
        line_num = content[: match.start()].count("\n") + 1

        # Skip private/internal names
        if base_name.startswith("_"):
            continue

        # Get namespace at this line
        ns = namespace_at_line.get(line_num - 1, "")

        # Build fully qualified name
        # When namespace is active, it IS the qualified path (don't add module)
        # Only use module_name as prefix when no namespace is active
        if ns:
            full_name = f"{ns}.{base_name}"
        elif module_name:
            full_name = f"{module_name}.{base_name}"
        else:
            full_name = base_name

        declarations.append(
            LeanDeclaration(
                name=full_name,
                kind=kind,
                module=module_name,
                signature=signature[:500] if signature else "",
                docstring=docstring[:1000] if docstring else None,
                file_path=str(file_path),
                line=line_num,
            )
        )

    return declarations
